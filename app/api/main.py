"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, chat, conversation, document
from app.config import settings
from app.db.sessions import close_db, init_db

# ── Logging ────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Personal LLM Assistant v2.0...")
    logger.info("LLM Provider: %s, Model: %s", settings.llm_provider, settings.llm_model)

    # Initialize database
    await init_db()

    # Ensure collections exist
    from app.api.dependencies import _get_dense_store
    dense = _get_dense_store()
    await dense.ensure_collection()
    await dense.ensure_image_collection()

    logger.info("Server ready on %s:%d", settings.host, settings.port)

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Save BM25 index
    from app.api.dependencies import _get_bm25_index
    bm25 = _get_bm25_index()
    if len(bm25) > 0:
        bm25.save(settings.bm25_index_path)

    await close_db()
    logger.info("Shutdown complete")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Personal LLM Assistant",
    description="Production RAG system with hybrid retrieval, query rewriting, and streaming responses",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router)
app.include_router(conversation.router)
app.include_router(document.router)
app.include_router(admin.router)


# ── Root ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "Personal LLM Assistant",
        "version": "2.0.0",
        "docs": "/docs",
    }
