from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .config import Settings, settings as default_settings
from .guardrails import validate_question
from .llm_clients import GeminiClient, LiteLLMGatewayClient, LLMClient
from .rag_prompts import build_prompt
from .retrieval import retrieve


@dataclass(frozen=True)
class GenerationResult:
	answer: str
	sources: List[Dict[str, Union[str, float]]]
	confidence: float


def build_llm_client(settings: Settings) -> LLMClient:
	"""Create the LLM client for the configured provider."""

	if settings.llm_provider == "litellm":
		return LiteLLMGatewayClient(
			base_url=settings.litellm_base_url,
			model=settings.llm_model,
			api_key=settings.litellm_api_key,
			timeout=settings.llm_timeout_seconds,
		)
	if settings.llm_provider == "gemini":
		if not settings.google_api_key:
			raise ValueError("GOOGLE_API_KEY is missing. Add it to your environment file.")
		return GeminiClient(api_key=settings.google_api_key, model=settings.llm_model)
	raise ValueError(
		f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'litellm' or 'gemini'."
	)


class RAGPipeline:
	def __init__(
		self,
		settings: Settings = default_settings,
		llm_client: Optional[LLMClient] = None,
	) -> None:
		self.settings = settings
		self._llm_client = llm_client

	def _get_llm_client(self) -> LLMClient:
		if self._llm_client is None:
			self._llm_client = build_llm_client(self.settings)
		return self._llm_client

	def _confidence(self, sources: List[Dict[str, Union[str, float]]]) -> float:
		if not sources:
			return 0.0
		return round(sum(float(source.get("score", 0.0)) for source in sources) / len(sources), 3)

	def answer(self, question: str) -> GenerationResult:
		validated_question = validate_question(question)
		retrieved_documents = retrieve(validated_question, top_k=self.settings.top_k)
		if not retrieved_documents:
			return GenerationResult(
				answer="I couldn't find enough information in the Rick & Morty knowledge base "
    "to answer this question.",
				sources=[],
				confidence=0.0,
			)

		prompt = build_prompt(validated_question, retrieved_documents)
		client = self._get_llm_client()
		answer_text = client.generate(prompt)
		return GenerationResult(
			answer=answer_text,
			sources=retrieved_documents,
			confidence=self._confidence(retrieved_documents),
		)
