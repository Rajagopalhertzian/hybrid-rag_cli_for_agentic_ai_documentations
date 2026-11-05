"""
Configuration management for hybrid-rag using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get project root from env var or current working directory."""
    # Allow explicit override via HYBRID_RAG_ROOT
    if root := os.environ.get("HYBRID_RAG_ROOT"):
        return Path(root).resolve()
    # Default to current working directory
    return Path.cwd().resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    project_root: Path = Field(default_factory=_get_project_root)
    data_dir: Path = Field(default_factory=lambda: _get_project_root() / "data")
    raw_data_dir: Path = Field(default_factory=lambda: _get_project_root() / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: _get_project_root() / "data" / "processed")
    eval_data_dir: Path = Field(default_factory=lambda: _get_project_root() / "data" / "eval")

    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-70b-versatile", validation_alias="GROQ_MODEL")

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    ollama_base_url: Optional[str] = Field(default=None, validation_alias="OLLAMA_BASE_URL")
    ollama_model: Optional[str] = Field(default=None, validation_alias="OLLAMA_MODEL")

    # -------------------------------------------------------------------------
    # Vector Database (Qdrant)
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_prefer_grpc: bool = Field(default=False, validation_alias="QDRANT_PREFER_GRPC")

    # -------------------------------------------------------------------------
    # Embedding Model
    # -------------------------------------------------------------------------
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1024, validation_alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=32, validation_alias="EMBEDDING_BATCH_SIZE")
    embedding_device: str = Field(default="cpu", validation_alias="EMBEDDING_DEVICE")

    # Optional reranker
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", validation_alias="RERANKER_MODEL")
    reranker_enabled: bool = Field(default=False, validation_alias="RERANKER_ENABLED")

    # -------------------------------------------------------------------------
    # Retrieval Settings
    # -------------------------------------------------------------------------
    retrieval_top_k: int = Field(default=10, validation_alias="RETRIEVAL_TOP_K")
    retrieval_vector_k: int = Field(default=50, validation_alias="RETRIEVAL_VECTOR_K")
    retrieval_bm25_k: int = Field(default=50, validation_alias="RETRIEVAL_BM25_K")
    retrieval_rrf_k: int = Field(default=60, validation_alias="RETRIEVAL_RRF_K")
    retrieval_rerank_top_k: int = Field(default=20, validation_alias="RETRIEVAL_RERANK_TOP_K")

    # -------------------------------------------------------------------------
    # Chunking Settings
    # -------------------------------------------------------------------------
    chunk_fixed_tokens: int = Field(default=512, validation_alias="CHUNK_FIXED_TOKENS")
    chunk_fixed_overlap: int = Field(default=50, validation_alias="CHUNK_FIXED_OVERLAP")
    chunk_heading_max_tokens: int = Field(default=1024, validation_alias="CHUNK_HEADING_MAX_TOKENS")

    # -------------------------------------------------------------------------
    # Fetching Settings
    # -------------------------------------------------------------------------
    fetch_rate_limit: float = Field(default=2.0, validation_alias="FETCH_RATE_LIMIT")
    fetch_timeout: float = Field(default=30.0, validation_alias="FETCH_TIMEOUT")
    fetch_max_retries: int = Field(default=3, validation_alias="FETCH_MAX_RETRIES")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    eval_questions_file: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "eval" / "questions.yaml",
        validation_alias="EVAL_QUESTIONS_FILE",
    )
    eval_results_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "eval" / "results",
        validation_alias="EVAL_RESULTS_DIR",
    )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(default="json", validation_alias="LOG_FORMAT")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings