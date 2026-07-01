from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from ..config import settings

API_BASE = "https://rickandmortyapi.com/api"
CHARACTERS = "character"
EPISODES = "episode"
LOCATIONS = "location"


@dataclass(frozen=True)
class Document:
    """Simple normalized document representation.

    Fields: `id`, `title`, `content`, `source`.
    """

    id: str
    title: str
    content: str
    source: str | None = None


def fetch_all(resource: str, retry: int = 3, pause: float = 0.5) -> List[dict]:
    """
    Fetch every page for a given Rick & Morty resource.

    Why: The API is paginated; this utility centralises pagination handling.
    Why here: This file is the data ingestion pipeline and needs raw API data.
    Assumptions: The API follows the documented `info.next` pagination field.
    """

    url = f"{API_BASE}/{resource}"
    items: List[dict] = []

    while url:
        for attempt in range(1, retry + 1):
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                break
            if attempt == retry:
                resp.raise_for_status()
            time.sleep(pause * attempt)

        data = resp.json()
        results = data.get("results") or []
        items.extend(results)
        url = data.get("info", {}).get("next")

    return items


def normalize_character(raw: dict) -> Document:
    """
    Convert a single raw character JSON into a Document.

    Why: Normalisation decouples downstream storage and embedding code from API shape.
    Why here: Ingest module owns transforming API records to the local document model.
    Assumptions: Fields such as `name`, `status`, `species` exist on the API object.
    """

    char_id = str(raw.get("id"))
    title = raw.get("name", "")
    parts = [
        f"Name: {raw.get('name', '')}",
        f"Status: {raw.get('status', '')}",
        f"Species: {raw.get('species', '')}",
        f"Type: {raw.get('type', '')}",
        f"Gender: {raw.get('gender', '')}",
        f"Created: {raw.get('created', '')}",
        f"Origin: {raw.get('origin', {}).get('name', '')}",
        f"Location: {raw.get('location', {}).get('name', '')}",
        f"Episodes: {len(raw.get('episode', []) )}",
        f"URL: {raw.get('url', '')}",
    ]

    content = "\n".join(part for part in parts if part)
    return Document(id=f"character:{char_id}", title=title, content=content, source=raw.get("url"))


def normalize_episode(raw: dict) -> Document:
    """
    Convert a single raw episode JSON into a Document.

    Why: Episodes have different fields; keep normalisation local to ingest.
    Why here: This file prepares the canonical documents stored locally.
    Assumptions: `name`, `air_date`, `episode`, and `characters` exist.
    """

    ep_id = str(raw.get("id"))
    title = raw.get("name", "")
    parts = [
        f"Name: {raw.get('name', '')}",
        f"Air date: {raw.get('air_date', '')}",
        f"Episode code: {raw.get('episode', '')}",
        f"Created: {raw.get('created', '')}",
        f"Characters: {len(raw.get('characters', []))}",
        f"URL: {raw.get('url', '')}",
    ]

    content = "\n".join(part for part in parts if part)
    return Document(id=f"episode:{ep_id}", title=title, content=content, source=raw.get("url"))


def normalize_location(raw: dict) -> Document:
    """
    Convert a single raw location JSON into a Document.

    Why: Locations include dimension/type information used in retrieval later.
    Why here: Normalisation is part of ingestion responsibilities.
    Assumptions: `name`, `type`, `dimension`, and `residents` exist.
    """

    loc_id = str(raw.get("id"))
    title = raw.get("name", "")
    parts = [
        f"Name: {raw.get('name', '')}",
        f"Type: {raw.get('type', '')}",
        f"Dimension: {raw.get('dimension', '')}",
        f"Created: {raw.get('created', '')}",
        f"Residents: {len(raw.get('residents', []))}",
        f"URL: {raw.get('url', '')}",
    ]

    content = "\n".join(part for part in parts if part)
    return Document(id=f"location:{loc_id}", title=title, content=content, source=raw.get("url"))


def init_db(db_path: Path) -> None:
    """
    Create a SQLite database and the `documents` table if it does not exist.

    Why: Persisting raw documents enables reproducible local retrieval and inspection.
    Why here: The ingest module owns creating the local knowledge store.
    Assumptions: Caller can write to `db_path`'s parent directory.
    """

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                source TEXT
            )
            """
        )
        # Create an FTS5 virtual table for lexical search (hybrid retrieval).
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id, title, content, tokenize = 'porter'
            )
            """
        )
        conn.commit()


def insert_documents(conn: sqlite3.Connection, documents: Iterable[Document]) -> None:
    """
    Insert or replace documents into the SQLite `documents` table.

    Why: Simple persistence; using INSERT OR REPLACE keeps the ingest idempotent.
    Why here: Writing to the DB is part of ingestion responsibilities.
    Assumptions: `conn` references a valid SQLite connection with `documents` table.
    """

    rows = []
    for doc in documents:
        doc_type, _, doc_id = doc.id.partition(":")
        rows.append((doc.id, doc_type or "unknown", doc.title, doc.content, doc.source))

    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO documents (id, type, title, content, source) VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    # Maintain FTS table: remove any existing entries for these ids and insert fresh rows.
    for doc in documents:
        cur.execute("DELETE FROM documents_fts WHERE id = ?", (doc.id,))
        cur.execute(
            "INSERT INTO documents_fts (id, title, content) VALUES (?, ?, ?)",
            (doc.id, doc.title, doc.content),
        )

    conn.commit()


def generate_embeddings(documents: Iterable[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64) -> tuple[np.ndarray, List[str]]:
    """
    Generate dense embeddings for the provided documents using SentenceTransformers.

    Why: Embeddings are required for later similarity search and RAG.
    Why here: This file is responsible for producing the local knowledge artefacts.
    Assumptions: The `sentence-transformers` package is installed and model can be downloaded.
    """

    model = SentenceTransformer(model_name)
    texts: List[str] = []
    ids: List[str] = []
    for doc in documents:
        # Use title + content to give the embedding stronger signal.
        texts.append((doc.title or "") + "\n" + doc.content)
        ids.append(doc.id)

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    arr = np.asarray(embeddings, dtype=np.float32)
    return arr, ids


def save_embeddings(embeddings: np.ndarray, ids: List[str], embeddings_path: Path, ids_path: Path) -> None:
    """
    Persist embeddings to disk as `.npy` and write ids mapping as JSON.

    Why: Storing embeddings locally avoids recomputation and keeps a simple vector store.
    Why here: Generation and saving of embeddings belong to ingestion responsibilities.
    Assumptions: Caller has write access to the target paths.
    """

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    with ids_path.open("w", encoding="utf-8") as fh:
        json.dump(ids, fh, ensure_ascii=False)


def save_metadata(embeddings: np.ndarray, model_name: str, metadata_path: Path, document_count: int | None = None) -> None:
    """
    Save simple metadata for the embeddings (model, dimension, created_at).
    """

    metadata = {
        "model": model_name,
        "embedding_model": model_name,
        "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "document_count": document_count,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def ingest_all(db_path: Path, embeddings_path: Path, ids_path: Path, model_name: str | None = None) -> None:
    """
    End-to-end ingestion: fetch, normalise, persist, embed, and save.

    Why: Convenience orchestration for local setup and CI smoke tests.
    Why here: This file is the ingestion entrypoint and coordinates the functions above.
    Assumptions: Network access to the public API and sufficient disk space.
    """

    # Resolve model name from settings if not provided
    model_name = model_name or settings.embedding_model

    # Fetch raw data
    print("Fetching characters...")
    raw_characters = fetch_all(CHARACTERS)
    print(f"Fetched {len(raw_characters)} characters.")

    print("Fetching episodes...")
    raw_episodes = fetch_all(EPISODES)
    print(f"Fetched {len(raw_episodes)} episodes.")

    print("Fetching locations...")
    raw_locations = fetch_all(LOCATIONS)
    print(f"Fetched {len(raw_locations)} locations.")

    # Normalize
    characters = [normalize_character(r) for r in raw_characters]
    episodes = [normalize_episode(r) for r in raw_episodes]
    locations = [normalize_location(r) for r in raw_locations]

    all_docs = [*characters, *episodes, *locations]

    # Persist documents using a context manager for the DB connection
    print("Saving knowledge base to SQLite...")
    init_db(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        insert_documents(conn, all_docs)
    print(f"Saved {len(all_docs)} documents to {db_path}")

    # Generate and save embeddings
    print("Generating embeddings...")
    embeddings, ids = generate_embeddings(all_docs, model_name=model_name)
    print("Saving artifacts...")
    save_embeddings(embeddings, ids, embeddings_path, ids_path)
    # Save metadata next to embeddings (include model and document count)
    metadata_path = ids_path.parent / "metadata.json"
    save_metadata(embeddings, model_name, metadata_path, document_count=len(all_docs))
    print("Ingestion completed successfully.")


if __name__ == "__main__":
    # Example local run with sensible defaults
    base = Path(__file__).resolve().parents[2]
    db_file = base / "data" / "oracle.db"
    embeddings_file = base / "data" / "embeddings.npy"
    ids_file = base / "data" / "embeddings.ids.json"
    ingest_all(db_file, embeddings_file, ids_file)

