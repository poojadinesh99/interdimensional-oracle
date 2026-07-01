from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Union
import traceback
from .config import settings

from .rag_pipeline import RAGPipeline


class ChatRequest(BaseModel):
	question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
	answer: str
	sources: List[Dict[str, Union[str, float]]]
	confidence: float


app = FastAPI(title=settings.app_name)

app.add_middleware(
	CORSMiddleware,
	# Allow all origins for local development.
	# In production this should be restricted.
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)
pipeline = RAGPipeline(settings=settings)


@app.get("/api/health")
def health() -> dict[str, str]:
	return {
		"status": "ok",
		"provider": settings.llm_provider,
		"model": settings.llm_model,
	}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
	try:
		result = pipeline.answer(payload.question)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		# Log full traceback to diagnose 500s during API usage.
		traceback.print_exc()
		raise HTTPException(
			status_code=500,
			detail=str(exc)
		) from exc


	return ChatResponse(answer=result.answer, sources=result.sources, confidence=result.confidence)
