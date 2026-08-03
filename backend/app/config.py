import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    # 5000 collides with macOS's AirPlay Receiver (ControlCenter), which
    # squats on it by default and silently swallows requests.
    PORT = int(os.getenv("PORT", "5050"))
    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

    # "local" composes an answer from the retrieval corpus, no API calls or
    # cost. "ollama" generates with a local model via Ollama, also free.
    # "anthropic" calls the Claude API for a generated answer (paid).
    ORACLE_MODE = os.getenv("ORACLE_MODE", "local")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    ORACLE_MAX_TOKENS = int(os.getenv("ORACLE_MAX_TOKENS", "300"))

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    CORPUS_PATH = DATA_DIR / "corpus.json"
    VOCAB_PATH = DATA_DIR / "vocab.json"
    EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
