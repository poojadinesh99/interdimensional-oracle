"""Loads the indexed oracle corpus and retrieves the most relevant snippets
for a given question using cosine similarity over bag-of-words vectors."""
from __future__ import annotations

import json

import numpy as np

from backend.app.config import Config
from backend.app.data.ingest import vectorize


class RetrievalIndex:
    def __init__(self, corpus_path=None, vocab_path=None, embeddings_path=None):
        self.corpus_path = corpus_path or Config.CORPUS_PATH
        self.vocab_path = vocab_path or Config.VOCAB_PATH
        self.embeddings_path = embeddings_path or Config.EMBEDDINGS_PATH
        self._entries = None
        self._vocab = None
        self._embeddings = None

    def _load(self) -> None:
        if self._entries is not None:
            return
        if not (self.corpus_path.exists() and self.vocab_path.exists() and self.embeddings_path.exists()):
            raise FileNotFoundError(
                "Oracle index not found. Run `python -m backend.app.data.ingest` first."
            )
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self._entries = json.load(f)
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self._vocab = json.load(f)
        self._embeddings = np.load(self.embeddings_path)

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        self._load()
        query_vec = vectorize(question, self._vocab)
        if not np.any(query_vec):
            # No vocabulary overlap: fall back to a few varied snippets
            # rather than nonsense-ranked noise.
            return [entry["text"] for entry in self._entries[:top_k]]

        scores = self._embeddings @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._entries[i]["text"] for i in top_indices]


_index = RetrievalIndex()


def retrieve(question: str, top_k: int | None = None) -> list[str]:
    return _index.retrieve(question, top_k or Config.RETRIEVAL_TOP_K)
