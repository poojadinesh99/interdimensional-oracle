"""Builds a bag-of-words vocabulary and embedding matrix from corpus.json.

Run as a module from the repo root:
    python -m backend.app.data.ingest
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np

from backend.app.config import Config

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(documents: list[str]) -> dict[str, int]:
    counts = Counter()
    for doc in documents:
        counts.update(set(tokenize(doc)))
    vocab = {word: idx for idx, (word, _) in enumerate(counts.most_common())}
    return vocab


def vectorize(text: str, vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in tokenize(text):
        idx = vocab.get(token)
        if idx is not None:
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def build(corpus_path=None, vocab_path=None, embeddings_path=None) -> None:
    corpus_path = corpus_path or Config.CORPUS_PATH
    vocab_path = vocab_path or Config.VOCAB_PATH
    embeddings_path = embeddings_path or Config.EMBEDDINGS_PATH

    with open(corpus_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    texts = [entry["text"] for entry in entries]
    vocab = build_vocab(texts)
    embeddings = np.stack([vectorize(text, vocab) for text in texts])

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    np.save(embeddings_path, embeddings)

    print(f"Indexed {len(entries)} entries, vocab size {len(vocab)}")


if __name__ == "__main__":
    build()
