from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from .config import Settings, settings as default_settings
from .guardrails import validate_question
from .rag_prompts import build_prompt
from .retrieval import retrieve


try:
	from google import genai
except ImportError:
	genai = None


@dataclass(frozen=True)
class GenerationResult:
	answer: str
	sources: List[Dict[str, Union[str, float]]]
	confidence: float


class GeminiClient:
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


class RAGPipeline:
	def __init__(
		self,
		settings: Settings = default_settings,
		llm_client: Optional[GeminiClient] = None,
	) -> None:
		self.settings = settings
		self._llm_client = llm_client

	def _get_llm_client(self) -> GeminiClient:
		if self._llm_client is not None:
			return self._llm_client
		if not self.settings.google_api_key:
			raise ValueError("GOOGLE_API_KEY is missing. Add it to your environment file.")
		self._llm_client = GeminiClient(api_key=self.settings.google_api_key, model=self.settings.llm_model)
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
