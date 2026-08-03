import ollama
import pytest

from backend.app import oracle as oracle_module
from backend.app.config import Config


def test_ask_local_returns_a_corpus_snippet(monkeypatch):
    monkeypatch.setattr(Config, "ORACLE_MODE", "local")
    answer = oracle_module.ask("Should I take the leap and start my own business?")
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_ask_anthropic_requires_api_key(monkeypatch):
    monkeypatch.setattr(Config, "ORACLE_MODE", "anthropic")
    monkeypatch.setattr(Config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(oracle_module, "_client", None)
    with pytest.raises(oracle_module.OracleError):
        oracle_module.ask("Should I take the leap and start my own business?")


def test_ask_ollama_returns_generated_text(monkeypatch):
    monkeypatch.setattr(Config, "ORACLE_MODE", "ollama")

    fake_response = ollama.ChatResponse(
        message=ollama.Message(role="assistant", content="Walk toward the fear, not away from it.")
    )
    monkeypatch.setattr(ollama, "chat", lambda **kwargs: fake_response)

    answer = oracle_module.ask("Should I take the leap and start my own business?")
    assert answer == "Walk toward the fear, not away from it."


def test_ask_ollama_wraps_connection_errors(monkeypatch):
    monkeypatch.setattr(Config, "ORACLE_MODE", "ollama")

    def raise_connection_error(**kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(ollama, "chat", raise_connection_error)

    with pytest.raises(oracle_module.OracleError):
        oracle_module.ask("Should I take the leap and start my own business?")
