"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Project paths ──────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    chroma_db_path: str = str(project_root / "data" / "chroma_db")
    sqlite_db_path: str = str(project_root / "data" / "app.db")

    # ── LLM Provider ───────────────────────────────────────────
    llm_provider: Literal["tongyi", "deepseek", "zhipu"] = "deepseek"
    llm_model: str = "deepseek-v4-pro"          # generation
    llm_model_cheap: str = "deepseek-chat"       # for query rewriting (cheaper)
    embedding_provider: Literal["tongyi", "deepseek", "zhipu"] = "tongyi"

    # ── API Keys ───────────────────────────────────────────────
    tongyi_api_key: str = ""
    tongyi_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    tongyi_embedding_model: str = "text-embedding-v3"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    zhipu_api_key: str = ""
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"

    # ── Retrieval ──────────────────────────────────────────────
    retrieval_top_k: int = 20           # initial retrieval per branch
    rerank_top_k: int = 5               # after cross-encoder
    rrf_k: int = 60                     # RRF constant
    bm25_index_path: str = str(project_root / "data" / "bm25_index.pkl")

    # ── Chunking ───────────────────────────────────────────────
    child_chunk_size: int = 200
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 800
    parent_chunk_overlap: int = 25

    # ── Query Rewriting ────────────────────────────────────────
    rewrite_history_turns: int = 6

    # ── CrossEncoder ───────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"

    # ── Server ─────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # ── Rate Limiting ──────────────────────────────────────────
    rate_limit_tokens: int = 60      # requests
    rate_limit_window: int = 60      # seconds

    # ── Logging ────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Global singleton
settings = Settings()
