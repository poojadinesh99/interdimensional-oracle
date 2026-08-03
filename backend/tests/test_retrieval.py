import numpy as np
import pytest

from backend.app.data import ingest
from backend.app.retrieval import RetrievalIndex


@pytest.fixture
def index(tmp_path):
    corpus = [
        {"id": "a", "text": "The river finds a new path around every stone in its way."},
        {"id": "b", "text": "Patience lets the slow glacier carve canyons over time."},
        {"id": "c", "text": "Courage means walking toward what frightens you."},
    ]
    corpus_path = tmp_path / "corpus.json"
    vocab_path = tmp_path / "vocab.json"
    embeddings_path = tmp_path / "embeddings.npy"

    import json

    corpus_path.write_text(json.dumps(corpus))
    ingest.build(corpus_path=corpus_path, vocab_path=vocab_path, embeddings_path=embeddings_path)

    return RetrievalIndex(corpus_path=corpus_path, vocab_path=vocab_path, embeddings_path=embeddings_path)


def test_retrieve_returns_top_k(index):
    results = index.retrieve("Tell me about rivers and stones", top_k=2)
    assert len(results) == 2
    assert any("river" in r.lower() for r in results)


def test_retrieve_ranks_best_match_first(index):
    results = index.retrieve("I need courage to face what scares me", top_k=1)
    assert "courage" in results[0].lower() or "frightens" in results[0].lower()


def test_missing_index_raises(tmp_path):
    missing = RetrievalIndex(
        corpus_path=tmp_path / "nope.json",
        vocab_path=tmp_path / "nope-vocab.json",
        embeddings_path=tmp_path / "nope.npy",
    )
    with pytest.raises(FileNotFoundError):
        missing.retrieve("anything")
