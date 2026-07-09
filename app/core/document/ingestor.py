"""End-to-end document ingestion pipeline.

parse → chunk → embed → store in Milvus + BM25 index
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Sequence

from app.config import settings
from app.core.document.chunker import ChunkGroup, ParentChildChunker
from app.core.document.embedder import embed_batch
from app.core.document.parser import parse_document
from app.core.llm.base import EmbeddingProvider
from app.core.llm.registry import get_embedding_provider

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Orchestrates the full ingestion pipeline.

    Usage::

        ingestor = DocumentIngestor(milvus_store, bm25_index)
        await ingestor.ingest_file(Path("lecture/DMV_01_Intro.pdf"))
    """

    def __init__(self, milvus_store, bm25_index):
        self.milvus = milvus_store
        self.bm25 = bm25_index
        self.chunker = ParentChildChunker(
            parent_size=settings.parent_chunk_size,
            child_size=settings.child_chunk_size,
            parent_overlap=settings.parent_chunk_overlap,
            child_overlap=settings.child_chunk_overlap,
        )
        self._embedding_provider: EmbeddingProvider | None = None

    async def _get_embedder(self) -> EmbeddingProvider:
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    async def ingest_file(self, filepath: Path) -> list[ChunkGroup]:
        """Ingest a single file into both Milvus and BM25 index.

        Returns the ChunkGroups for downstream use.
        """
        # 1. Parse
        parsed = parse_document(filepath)
        logger.info("Parsed: %s (%d chars)", parsed.filename, len(parsed.text))

        # 2. Chunk
        groups = self.chunker.chunk(parsed.text, source_metadata=parsed.metadata or {})
        logger.info("Chunked into %d parent groups, %d total children",
                     len(groups), sum(len(g.children) for g in groups))

        # 3. Collect all child texts for embedding
        child_texts = [c.content for g in groups for c in g.children]
        child_ids = [c.id for g in groups for c in g.children]

        # 4. Embed
        embedder = await self._get_embedder()
        embeddings = await embed_batch(child_texts, embedder)

        # 5. Store in Milvus (dense)
        # Build metadata list: each child gets its parent_id + source info
        metadata_list = [
            {
                "child_id": cid,
                "parent_id": cid.rsplit("_c_", 1)[0],
                "source": filepath.name,
            }
            for cid in child_ids
        ]
        await self.milvus.insert(child_ids, embeddings, metadata_list)
        logger.info("Inserted %d vectors into Milvus", len(child_ids))

        # 6. Store in BM25 (sparse)
        self.bm25.index_documents(child_texts, child_ids)

        # 7. Store parent chunks (for context retrieval later)
        # We'll store parents in the BM25 index's metadata store
        for g in groups:
            self.bm25.store_parent(g.parent_id, g.parent_content, g.metadata)

        return groups

    async def ingest_directory(self, directory: Path, glob_pattern: str = "*.pdf") -> int:
        """Ingest all matching files in a directory. Returns count of files processed."""
        files = sorted(directory.glob(glob_pattern))
        logger.info("Ingesting %d files from %s", len(files), directory)

        count = 0
        for filepath in files:
            try:
                await self.ingest_file(filepath)
                count += 1
            except Exception:
                logger.exception("Failed to ingest: %s", filepath.name)
                # Continue with remaining files

        logger.info("Ingestion complete: %d/%d files succeeded", count, len(files))
        return count
