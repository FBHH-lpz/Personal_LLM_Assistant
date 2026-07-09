"""FastAPI dependencies — injected singletons and sessions."""

from __future__ import annotations

from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.graph.graph import build_rag_graph
from app.core.llm.base import ChatModel, EmbeddingProvider
from app.core.llm.registry import (
    get_chat_model,
    get_cheap_chat_model,
    get_embedding_provider,
)
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker
from app.db.sessions import get_session


# ── Singleton instances (lazy init) ───────────────────────────

@lru_cache(maxsize=1)
def _get_bm25_index() -> BM25Index:
    idx = BM25Index()
    idx.load(settings.bm25_index_path)
    return idx


@lru_cache(maxsize=1)
def _get_dense_store() -> DenseStore:
    return DenseStore(db_path=settings.milvus_db_path)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
    )


@lru_cache(maxsize=1)
def _get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense_store=_get_dense_store(),
        bm25_index=_get_bm25_index(),
        embed_provider=get_embedding_provider(),
    )


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Build and return the compiled RAG graph."""
    return build_rag_graph(
        chat_model=get_chat_model(),
        cheap_model=get_cheap_chat_model(),
        retriever=_get_hybrid_retriever(),
        reranker=_get_reranker(),
    )


# ── Public dependency functions ────────────────────────────────

def get_graph():
    """Inject the compiled RAG graph."""
    return _get_rag_graph()


def get_chat():
    """Inject the primary chat model."""
    return get_chat_model()


def get_embedder():
    """Inject the embedding provider."""
    return get_embedding_provider()


def get_retriever():
    """Inject the hybrid retriever."""
    return _get_hybrid_retriever()


def get_reranker_dep():
    """Inject the CrossEncoder reranker."""
    return _get_reranker()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Inject an async DB session."""
    async for session in get_session():
        yield session
