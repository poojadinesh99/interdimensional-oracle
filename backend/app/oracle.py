"""Generates the oracle's answer, via the local corpus, a local Ollama
model, or the Anthropic API, depending on ORACLE_MODE."""
from __future__ import annotations

import anthropic
import ollama

from backend.app.config import Config
from backend.app.prompts import SYSTEM_PROMPT, build_user_message
from backend.app.retrieval import retrieve

_client: anthropic.Anthropic | None = None


class OracleError(Exception):
    """Raised when the oracle cannot produce an answer."""


def _get_client() -> anthropic.Anthropic:
    global _client
    if not Config.ANTHROPIC_API_KEY:
        raise OracleError("ANTHROPIC_API_KEY is not configured.")
    if _client is None:
        _client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
    return _client


def ask(question: str) -> str:
    if Config.ORACLE_MODE == "anthropic":
        return _ask_anthropic(question)
    if Config.ORACLE_MODE == "ollama":
        return _ask_ollama(question)
    return _ask_local(question)


def _ask_local(question: str) -> str:
    """Composes an answer from the retrieval corpus. No API calls, no cost —
    the corpus entries are already written in the oracle's voice. Blending
    the top two matches instead of returning one fixed line means the same
    question rarely produces the exact same wording twice, since it depends
    on which pair of entries best matches this specific phrasing."""
    try:
        snippets = retrieve(question, top_k=2)
    except FileNotFoundError as exc:
        raise OracleError(str(exc)) from exc

    if not snippets:
        raise OracleError("The oracle has nothing to say yet.")
    return " ".join(snippets)


def _ask_ollama(question: str) -> str:
    context_snippets = retrieve(question)
    user_message = build_user_message(question, context_snippets)

    try:
        response = ollama.chat(
            model=Config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except ollama.ResponseError as exc:
        raise OracleError(f"The oracle is unreachable right now: {exc}") from exc
    except ConnectionError as exc:
        raise OracleError(
            "Can't reach the local Ollama server. Run `ollama serve` and make sure "
            f"`{Config.OLLAMA_MODEL}` is pulled."
        ) from exc

    answer = (response.message.content or "").strip()
    if not answer:
        raise OracleError("The oracle returned silence. Try asking again.")
    return answer


def _ask_anthropic(question: str) -> str:
    client = _get_client()
    context_snippets = retrieve(question)
    user_message = build_user_message(question, context_snippets)

    try:
        response = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=Config.ORACLE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise OracleError(f"The oracle is unreachable right now: {exc}") from exc

    text_blocks = [block.text for block in response.content if block.type == "text"]
    answer = "".join(text_blocks).strip()
    if not answer:
        raise OracleError("The oracle returned silence. Try asking again.")
    return answer
