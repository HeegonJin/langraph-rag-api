"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── llama.cpp server ──────────────────────────────────────────────────────────
LLAMA_CPP_BASE_URL: str = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
LLAMA_CPP_EMBED_BASE_URL: str = os.getenv("LLAMA_CPP_EMBED_BASE_URL", "http://localhost:8081/v1")
LLAMA_CPP_MODEL: str = os.getenv("LLAMA_CPP_MODEL", "default")
LLAMA_CPP_EMBED_MODEL: str = os.getenv("LLAMA_CPP_EMBED_MODEL", "default")
LLAMA_CPP_API_KEY: str = os.getenv("LLAMA_CPP_API_KEY", "no-key")

# ── Elasticsearch ─────────────────────────────────────────────────────────────
ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "rag-documents")

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Minimum cosine-similarity score for retrieved documents.
# Set to 0.0 to disable threshold filtering and rely on top-K only.
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))

# ── Docling ───────────────────────────────────────────────────────────────────
DOCLING_OCR_ENABLED: bool = os.getenv("DOCLING_OCR_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
DOCLING_TABLE_MODE: str = os.getenv("DOCLING_TABLE_MODE", "accurate")  # "accurate" or "fast"
DOCLING_TIMEOUT: int = int(os.getenv("DOCLING_TIMEOUT", "120"))

# ── LLM timeout ────────────────────────────────────────────────────────────────
# Seconds to wait for an LLM response before timing out.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "120"))

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Langfuse observability ────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "true").lower() in ("1", "true", "yes")

# ── Upload directory ──────────────────────────────────────────────────────────
UPLOAD_DIR: Path = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Sample data (auto-ingested on startup) ────────────────────────────────────
SAMPLE_DATA_DIR: Path = Path("sample_data")
