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

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

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
