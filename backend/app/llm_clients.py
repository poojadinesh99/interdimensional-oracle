"""LLM clients used by the RAG pipeline.

Two providers are supported, selected with LLM_PROVIDER:

- ``litellm``: calls a LiteLLM proxy (the model gateway) over its
  OpenAI-compatible ``/chat/completions`` endpoint. The gateway owns the
  provider keys, model aliases, retries, fallbacks and spend tracking, so the
  app only knows a gateway URL, a virtual key and a model alias.
- ``gemini``: calls Gemini directly with the google-genai SDK (original setup,
  kept for running without the gateway).
"""

from __future__ import annotations

from typing import Optional, Protocol

import httpx

try:
	from google import genai
except ImportError:
	genai = None


class LLMClient(Protocol):
	def generate(self, prompt: str) -> str: ...


class LLMGatewayError(RuntimeError):
	"""Raised when the LiteLLM gateway returns an error or an unusable response."""


class LiteLLMGatewayClient:
	"""Client for a LiteLLM proxy (OpenAI-compatible API).

	`model` is the alias defined in gateway/litellm_config.yaml (e.g.
	"oracle-default"), not a provider model id. Which real model serves the
	request, and what happens when it fails, is decided by the gateway.
	"""

	def __init__(
		self,
		base_url: str,
		model: str,
		api_key: Optional[str] = None,
		timeout: float = 60.0,
		http_client: Optional[httpx.Client] = None,
	) -> None:
		headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
		self._model = model
		self._http = http_client or httpx.Client(
			base_url=base_url.rstrip("/"),
			headers=headers,
			timeout=timeout,
		)

	def generate(self, prompt: str) -> str:
		try:
			response = self._http.post(
				"/chat/completions",
				json={
					"model": self._model,
					"messages": [{"role": "user", "content": prompt}],
					# Tag requests so they can be filtered in LiteLLM's spend logs.
					"metadata": {"tags": ["interdimensional-oracle", "rag-answer"]},
				},
			)
		except httpx.HTTPError as exc:
			raise LLMGatewayError(f"LiteLLM gateway is unreachable: {exc}") from exc

		if response.status_code >= 400:
			raise LLMGatewayError(
				f"LiteLLM gateway returned {response.status_code}: {response.text[:500]}"
			)

		try:
			content = response.json()["choices"][0]["message"]["content"]
		except (ValueError, KeyError, IndexError, TypeError) as exc:
			raise LLMGatewayError("LiteLLM gateway returned an unexpected response shape.") from exc
		return (content or "").strip()


class GeminiClient:
	"""Direct Gemini client (no gateway)."""

	def __init__(self, api_key: str, model: str) -> None:
		if genai is None:
			raise RuntimeError(
				"The google-genai package is not installed. Install requirements.txt first."
			)
		self._client = genai.Client(api_key=api_key)
		self._model = model

	def generate(self, prompt: str) -> str:
		response = self._client.models.generate_content(
			model=self._model,
			contents=prompt,
		)
		return (getattr(response, "text", None) or "").strip()
