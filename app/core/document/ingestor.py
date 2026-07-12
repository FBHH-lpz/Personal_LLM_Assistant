"""End-to-end document ingestion with structured VLM analysis and multi-vector indexing.

parse → chunk → (concurrent VLM for images) → embed → store (text + images)
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
    """Orchestrates ingestion with VLM image analysis and multi-vector indexing."""

    def __init__(self, dense_store, bm25_index):
        self.dense = dense_store
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
        """Ingest a single file with structured VLM image analysis.

        1. Parse PDF (text + detect images)
        2. Concurrent VLM: analyze all images in parallel (structured JSON)
        3. Cross-page merge: concatenate adjacent pages for full context
        4. Parent-child chunk text
        5. Embed and store text chunks in main collection
        6. Embed and store image descriptions in separate image collection
        7. Index everything in BM25
        """
        # ── 1. Parse ──────────────────────────────────────────
        parsed = parse_document(filepath)
        file_prefix = filepath.stem.replace(" ", "_")[:32]
        logger.info("Parsed: %s (%d chars, %d image pages)",
                     parsed.filename, len(parsed.text), len(parsed.image_pages))

        # ── 2. Concurrent VLM analysis ─────────────────────────
        image_descriptions: list[str] = []
        image_chunk_ids: list[str] = []
        image_bindings: dict[int, str] = {}  # page_num → image_chunk_id

        if parsed.image_pages:
            from app.core.llm.tongyi_vlm import (
                describe_images_concurrent,
                structured_desc_to_text,
            )

            # Build image paths list
            img_paths = [ip["image_path"] for ip in parsed.image_pages]

            logger.info("Analyzing %d images with VLM (concurrent, max %d)...",
                        len(img_paths), 5)

            # Concurrent VLM calls!
            results = await describe_images_concurrent(img_paths)

            for i, (img_info, desc) in enumerate(zip(parsed.image_pages, results)):
                page_num = img_info["page_number"]
                if desc and desc.get("type"):
                    # Convert structured JSON to searchable text
                    chunk_text = structured_desc_to_text(desc, filepath.name, page_num)
                    if chunk_text:
                        img_id = f"{file_prefix}_img_p{page_num}"
                        image_descriptions.append(chunk_text)
                        image_chunk_ids.append(img_id)
                        image_bindings[page_num] = img_id
                        logger.debug("  Page %d: %s described (%d chars)",
                                     page_num, desc.get("type"), len(chunk_text))
                    else:
                        logger.warning("  Page %d: empty description", page_num)
                else:
                    logger.warning("  Page %d: VLM failed or returned empty", page_num)

        # ── 3. Extract lecture number for metadata ───────────────
        import re
        lecture_num = ""
        match = re.search(r'DMV[_]?(\d+)', filepath.stem)
        if match:
            lecture_num = match.group(1)

        # ── 4. Build page → VLM description mapping ───────────
        page_descriptions: dict[int, str] = {}
        for page_num, desc_text in zip(image_bindings.keys(), image_descriptions):
            page_descriptions[page_num] = desc_text

        # ── 5. Text chunking ──────────────────────────────────
        lecture_tag = f"[课件{lecture_num}] " if lecture_num else ""
        groups = self.chunker.chunk(parsed.text, source_metadata=parsed.metadata or {})
        total_pages = max(parsed.page_count, 1)
        total_parents = len(groups)

        for i, g in enumerate(groups):
            g.parent_id = f"{file_prefix}_{g.parent_id}"
            if lecture_tag:
                g.parent_content = lecture_tag + g.parent_content

            # Append VLM image description to parent if same page
            est_page = int(i / max(total_parents, 1) * total_pages) + 1
            img_desc = page_descriptions.get(est_page, "")
            if img_desc:
                g.parent_content = f"{g.parent_content}\n\n[本页图表分析] {img_desc}"

            for c in g.children:
                c.parent_id = g.parent_id
                c.content = lecture_tag + c.content if lecture_tag else c.content
                c.id = f"{g.parent_id}_c_{c.id.split('_c_')[-1]}" if "_c_" in c.id else f"{g.parent_id}_c_0"
        logger.info("Chunked: %d parents, %d children, %d with image descs",
                     len(groups), sum(len(g.children) for g in groups),
                     sum(1 for p in page_descriptions.values() if p))

        # ── 6. Embed text chunks ──────────────────────────────
        child_texts = [c.content for g in groups for c in g.children]
        child_ids = [c.id for g in groups for c in g.children]

        embedder = await self._get_embedder()
        embeddings = await embed_batch(child_texts, embedder)

        # ── 7. Store text in main collection ──────────────────
        metadata_list = [
            {
                "child_id": cid,
                "parent_id": cid.rsplit("_c_", 1)[0],
                "source": filepath.name,
            }
            for cid in child_ids
        ]
        self.dense.insert(child_ids, embeddings, metadata_list)
        self.bm25.index_documents(child_texts, child_ids)

        # ── 7b. Store images in separate collection (independent retrieval path)
        if image_descriptions:
            img_embeddings = await embed_batch(image_descriptions, embedder)
            img_metadata = [
                {
                    "child_id": iid,
                    "source": filepath.name,
                    "type": "image_description",
                }
                for iid in image_chunk_ids
            ]
            self.dense.insert_images(image_chunk_ids, img_embeddings, img_metadata)
            self.bm25.index_documents(image_descriptions, image_chunk_ids)

        # ── 8. Store text parents (already contain VLM descriptions) ──
        for g in groups:
            self.bm25.store_parent(g.parent_id, g.parent_content, g.metadata)

        logger.info("Ingested %s: %d text chunks", filepath.name, len(child_ids))
        return groups

    async def ingest_directory(self, directory: Path, glob_pattern: str = "*.pdf") -> int:
        """Ingest all matching files in a directory."""
        files = sorted(directory.glob(glob_pattern))
        logger.info("Ingesting %d files from %s", len(files), directory)
        count = 0
        for filepath in files:
            try:
                await self.ingest_file(filepath)
                count += 1
            except Exception:
                logger.exception("Failed to ingest: %s", filepath.name)
        logger.info("Ingestion complete: %d/%d files", count, len(files))
        return count
