import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure `.env` is loaded when running the API locally (uvicorn reload / direct run).
try:
	from dotenv import load_dotenv

	# backend/.env sits one directory above this module: backend/app/config.py
	load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except Exception:
	# If python-dotenv isn't installed or file missing, fall back to real env vars.
	pass


@dataclass(frozen=True)
class Settings:
	app_name: str = os.getenv("APP_NAME", "Interdimensional Oracle")
	app_env: str = os.getenv("APP_ENV", "development")
	llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
	llm_model: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
	google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
	# LiteLLM gateway (used when LLM_PROVIDER=litellm; MODEL_NAME is then a gateway alias)
	litellm_base_url: str = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
	litellm_api_key: Optional[str] = os.getenv("LITELLM_API_KEY")
	llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
	database_path: Path = Path(os.getenv("DATABASE_PATH", "backend/data/oracle.db"))

	embedding_model: str = os.getenv(
		"EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
	)
	top_k: int = int(os.getenv("TOP_K", "3"))
	max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))


settings = Settings()
