#!/usr/bin/env python3
"""Batch document ingestion script.

Ingests all PDFs from the lecture/ directory into Milvus + BM25 index.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --dir lecture --pattern "*.pdf"
    python scripts/ingest_docs.py --file lecture/DMV_01_Intro.pdf
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.document.ingestor import DocumentIngestor
from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Batch document ingestion for RAG system")
    parser.add_argument("--dir", help="Directory containing documents to ingest")
    parser.add_argument("--file", help="Single file to ingest")
    parser.add_argument("--pattern", default="*.pdf", help="File pattern for directory mode")
    args = parser.parse_args()

    # ── Initialize storage ──────────────────────────────────

    logger.info("Initializing ChromaDB at %s", settings.chroma_db_path)
    dense = DenseStore(db_path=settings.chroma_db_path)
    await dense.ensure_collection()

    # Check existing count
    existing = await dense.count()
    logger.info("Existing vectors in Milvus: %d", existing)

    logger.info("Initializing BM25 index")
    bm25 = BM25Index()
    # Try to load existing index
    loaded = bm25.load(settings.bm25_index_path)
    if loaded:
        logger.info("Loaded existing BM25 index: %d documents", len(bm25))

    # ── Get embedding provider ──────────────────────────────

    embedder = get_embedding_provider()
    logger.info("Embedding provider ready: %s", settings.embedding_provider)

    # ── Ingest ──────────────────────────────────────────────

    ingestor = DocumentIngestor(
        dense_store=dense,
        bm25_index=bm25,
    )

    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            logger.error("File not found: %s", filepath)
            sys.exit(1)
        logger.info("Ingesting single file: %s", filepath.name)
        groups = await ingestor.ingest_file(filepath)
        count = sum(len(g.children) for g in groups)
        logger.info("Done: %d child chunks created", count)

    elif args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            logger.error("Directory not found: %s", directory)
            sys.exit(1)
        file_count = await ingestor.ingest_directory(directory, glob_pattern=args.pattern)
        logger.info("Done: %d files ingested", file_count)

    else:
        # Default: ingest all PDFs from lecture/
        lecture_dir = Path("lecture")
        if not lecture_dir.is_dir():
            logger.error("Default lecture/ directory not found. Use --dir or --file.")
            sys.exit(1)
        file_count = await ingestor.ingest_directory(lecture_dir, glob_pattern="*.pdf")
        logger.info("Done: %d files ingested from lecture/", file_count)

    # ── Save BM25 index ─────────────────────────────────────

    bm25.save(settings.bm25_index_path)
    logger.info("BM25 index saved to %s", settings.bm25_index_path)

    # ── Final stats ─────────────────────────────────────────

    final_count = await dense.count()
    logger.info("Final vector count: %d", final_count)
    logger.info("Final BM25 document count: %d", len(bm25))
    logger.info("Parent chunks stored: %d", len(bm25._parents) if hasattr(bm25, "_parents") else 0)


if __name__ == "__main__":
    asyncio.run(main())
