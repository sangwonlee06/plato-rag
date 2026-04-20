"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All configuration is driven by environment variables prefixed PLATO_RAG_."""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/plato_rag"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/plato_rag"

    # Embedding (OpenAI)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # Generation (Anthropic)
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-4-20250514"
    generation_max_tokens: int = 2048

    # Service
    api_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    bootstrap_enabled: bool = True
    bootstrap_manifest_path: Path = _PROJECT_ROOT / "data" / "corpus_seed.json"
    bootstrap_lock_id: int = 712_341_905
    bootstrap_http_timeout_seconds: float = 30.0
    bootstrap_max_chunk_tokens: int = 512
    bootstrap_min_chunk_tokens: int = 50
    bootstrap_overlap_tokens: int = 64

    model_config = {"env_prefix": "PLATO_RAG_"}
