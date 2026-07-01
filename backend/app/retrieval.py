"""Hybrid retrieval utilities.

This module implements a simple hybrid retriever that combines SQLite FTS5
keyword search with semantic similarity over precomputed embeddings.

Design goals:
- Keep functions single-responsibility.
- Use settings for defaults so the project stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

from .config import settings


_embedding_model: SentenceTransformer | None = None


@dataclass(frozen=True)
class DocumentRecord:
	"""A document loaded from the local SQLite store.

	Holds the minimal fields required by the rest of the system.
	"""

	id: str
	title: str
	content: str
	source: Optional[str]


def _load_documents(db_path: Path, ids: Iterable[str]) -> Mapping[str, DocumentRecord]:
	"""
	Load documents by id from the SQLite `documents` table.

	Returns a mapping id -> DocumentRecord.
	"""

	db_path = Path(db_path)
	placeholders = ",".join("?" for _ in ids) or "''"
	query = f"SELECT id, title, content, source FROM documents WHERE id IN ({placeholders})"

	with sqlite3.connect(str(db_path)) as conn:
		cur = conn.cursor()
		cur.execute(query, tuple(ids))
		rows = cur.fetchall()

	return {row[0]: DocumentRecord(id=row[0], title=row[1] or "", content=row[2] or "", source=row[3]) for row in rows}


def load_embeddings(embeddings_path: Path, ids_path: Path) -> Tuple[np.ndarray, List[str]]:
	"""Load embeddings matrix and the parallel list of ids.

	Returns (embeddings, ids) where embeddings is float32 ndarray shape (N, D).
	"""

	arr = np.load(str(embeddings_path))
	with open(ids_path, "r", encoding="utf-8") as fh:
		ids = json.load(fh)
	return np.asarray(arr, dtype=np.float32), list(ids)


def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
	"""Compute cosine similarities between query_vec (D,) and matrix (N,D).

	Returns a 1-D array of length N.
	"""

	# normalize to unit vectors to make similarity a dot product
	q = query_vec.astype(np.float32)
	q_norm = np.linalg.norm(q)
	if q_norm == 0.0:
		return np.zeros(matrix.shape[0], dtype=np.float32)
	q = q / q_norm

	m = matrix.astype(np.float32)
	m_norms = np.linalg.norm(m, axis=1)
	# avoid division by zero
	m_norms[m_norms == 0.0] = 1.0
	m = (m.T / m_norms).T

	sims = m.dot(q)
	return sims


def embed_query(text: str, model_name: Optional[str] = None) -> np.ndarray:
	"""Generate an embedding for `text` using SentenceTransformer.

	Uses `settings.embedding_model` when `model_name` is None so the project
	remains consistent with the ingest pipeline.
	"""

	model = get_embedding_model(model_name)
	vec = model.encode([text], show_progress_bar=False)[0]
	return np.asarray(vec, dtype=np.float32)


def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
	"""Load the SentenceTransformer once and reuse it for all queries.

	Why: Model loading is expensive, so caching avoids reloading on every query.
	Why here: This module owns query embedding generation and should manage the model lifecycle.
	Assumptions: The configured embedding model name stays stable during process lifetime.
	"""

	global _embedding_model

	desired_model = model_name or settings.embedding_model
	if _embedding_model is None:
		_embedding_model = SentenceTransformer(desired_model)
	return _embedding_model


def semantic_search(query: str, embeddings: np.ndarray, ids: List[str], top_k: int = 10, model_name: Optional[str] = None) -> List[Tuple[str, float, int]]:
	"""Return ranked list of (id, score, rank) from dense embeddings.

	Score is cosine similarity in [-1,1]. Rank is 1-based position.
	"""

	qvec = embed_query(query, model_name=model_name)
	sims = _cosine_similarities(qvec, embeddings)
	order = np.argsort(-sims)
	results: List[Tuple[str, float, int]] = []
	for rank, idx in enumerate(order[:top_k], start=1):
		results.append((ids[int(idx)], float(sims[int(idx)]), rank))
	return results


def fts_search(db_path: Path, query: str, top_k: int = 10) -> List[Tuple[str, int]]:
	"""Run a simple FTS5 MATCH query and return ordered ids with rank positions.

	We return (id, rank) pairs; the calling code will convert ranks to RRF scores.
	"""

	results: List[Tuple[str, int]] = []
	with sqlite3.connect(str(db_path)) as conn:
		cur = conn.cursor()
		# FTS5 MATCH doesn't always accept bound parameters consistently across SQLite versions.
		# Safest option here is to interpolate a quoted query string.
		escaped = query.replace("\"", "\"\"")
		fts_query = f"\"{escaped}\""
		cur.execute(
			"""
			SELECT id, bm25(documents_fts) AS rank
			FROM documents_fts
			WHERE documents_fts MATCH ?
			ORDER BY rank
			LIMIT ?
			""",
			(fts_query, top_k),
		)
		rows = cur.fetchall()
	for rank, row in enumerate(rows, start=1):
		results.append((row[0], rank))
	return results


def rrf_merge(sem_results: List[Tuple[str, float, int]], fts_results: List[Tuple[str, int]], top_k: int = 10, rrf_k: int = 60) -> List[Tuple[str, float]]:
	"""Reciprocal Rank Fusion of two ranked lists.

	`sem_results` items are (id, score, rank). `fts_results` items are (id, rank).
	Returns list of (id, combined_score) sorted desc.
	"""

	scores: dict[str, float] = {}

	for _id, _score, rank in sem_results:
		scores.setdefault(_id, 0.0)
		scores[_id] += 1.0 / (rrf_k + rank)

	for _id, rank in fts_results:
		scores.setdefault(_id, 0.0)
		scores[_id] += 1.0 / (rrf_k + rank)

	merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
	# Normalize combined scores to [0,1] for downstream consumption
	max_score = merged[0][1] if merged else 1.0
	return [(_id, float(score / max_score)) for _id, score in merged]


def retrieve(
	query: str,
	top_k: int = 10,
	db_path: Optional[Path] = None,
	embeddings_path: Optional[Path] = None,
	ids_path: Optional[Path] = None,
) -> List[Dict[str, Union[str, float]]]:
	"""Top-level retrieval function used by the RAG pipeline.

	Returns a list of dictionaries with keys: `id`, `title`, `content`, `source`, `score`.
	"""

	db_path = Path(db_path or settings.database_path)
	base = db_path.parent
	embeddings_path = Path(embeddings_path or (base / "embeddings.npy"))
	ids_path = Path(ids_path or (base / "embeddings.ids.json"))

	# Prefer embeddings/db artefacts in the project-level `backend/data`.
	# This keeps runtime compatible with how ingest.py persists files.
	project_data_dir = Path(__file__).resolve().parents[1] / "data"
	fallback_data_dir = Path(__file__).resolve().parent / "data"

	if not db_path.exists():
		if (project_data_dir / "oracle.db").exists():
			db_path = project_data_dir / "oracle.db"
		else:
			db_path = fallback_data_dir / "oracle.db"

	if not embeddings_path.exists():
		if (project_data_dir / "embeddings.npy").exists():
			embeddings_path = project_data_dir / "embeddings.npy"
		else:
			embeddings_path = fallback_data_dir / "embeddings.npy"

	if not ids_path.exists():
		if (project_data_dir / "embeddings.ids.json").exists():
			ids_path = project_data_dir / "embeddings.ids.json"
		else:
			ids_path = fallback_data_dir / "embeddings.ids.json"

	# Load dense embeddings and ids
	embeddings, ids = load_embeddings(embeddings_path, ids_path)

	# Semantic search (top_k * 2 to give room for fusion)
	sem_top = max(top_k * 2, top_k)
	sem_results = semantic_search(query, embeddings, ids, top_k=sem_top, model_name=None)

	# FTS search
	fts_results = fts_search(db_path, query, top_k=top_k * 2)

	# Merge via RRF
	merged = rrf_merge(sem_results, fts_results, top_k=top_k)

	# Load document fields for returned ids
	ids_order = [doc_id for doc_id, _ in merged]
	docs_map = _load_documents(db_path, ids_order)

	results: List[dict] = []
	for doc_id, score in merged:
		doc = docs_map.get(doc_id)
		if not doc:
			# skip missing documents
			continue
		results.append({
			"id": doc.id,
			"title": doc.title,
			"content": doc.content,
			"source": doc.source,
			"score": score,
		})

	return results
