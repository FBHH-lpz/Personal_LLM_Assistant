"""Hybrid Retriever — BM25 + Dense combined with RRF fusion.

This is the core retrieval engine. It runs BM25 and dense search in
parallel, then merges results via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import Settings, settings
from app.core.llm.base import EmbeddingProvider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """A retrieved document chunk with metadata."""
    child_id: str
    parent_id: str
    content: str         # resolved from BM25 or parent store
    source: str = ""
    score: float = 0.0


class HybridRetriever:
    """BM25 + Dense hybrid retrieval with RRF fusion.

    Usage::

        retriever = HybridRetriever(dense_store, bm25_index, embedder)
        docs = await retriever.search("Transformer架构是什么", top_k=20)
    """

    def __init__(
        self,
        dense_store: DenseStore,
        bm25_index: BM25Index,
        embed_provider: EmbeddingProvider,
        settings_obj: Settings | None = None,
    ):
        self.dense = dense_store
        self.bm25 = bm25_index
        self.embedder = embed_provider
        self.s = settings_obj or settings

    async def search(
        self,
        query: str,
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedDoc]:
        """Run hybrid search and return merged, ranked results.

        Args:
            query: The search query (already rewritten if multi-turn).
            top_k: Number of results to return after fusion.
            rrf_k: RRF constant (default 60 from empirical studies).

        Returns:
            List of RetrievedDoc sorted by RRF score descending.
        """
        # 1. Get query embedding
        query_embedding = await self.embedder.aembed_single(query)

        # 2. Run BM25 + Dense in parallel
        loop = asyncio.get_event_loop()

        # BM25 is CPU-bound: run in executor
        bm25_task = loop.run_in_executor(
            None,
            lambda: self.bm25.search(query, top_k=top_k),
        )

        # Dense (text) + Dense (images) are both async
        dense_task = self.dense.search(query_embedding, top_k=top_k)
        image_task = self.dense.search_images(query_embedding, top_k=top_k // 2)

        bm25_results, dense_results, image_results = await asyncio.gather(
            bm25_task, dense_task, image_task,
        )

        logger.debug("BM25: %d, Dense: %d, Images: %d results",
                     len(bm25_results), len(dense_results), len(image_results))

        # 3. RRF Fusion (three-way: BM25 + text dense + image dense)
        merged = self._rrf_fusion_3way(
            bm25_results, dense_results, image_results,
            k=rrf_k, top_k=top_k,
        )

        # 4. Resolve parent content for each result
        docs: list[RetrievedDoc] = []
        for child_id, rrf_score in merged:
            parent_ctx = self.bm25.get_parent(
                child_id.rsplit("_c_", 1)[0] if "_c_" in child_id else child_id
            )
            content = parent_ctx[0] if parent_ctx else ""
            source = parent_ctx[1].get("source", "") if parent_ctx else ""

            docs.append(RetrievedDoc(
                child_id=child_id,
                parent_id=child_id.rsplit("_c_", 1)[0] if "_c_" in child_id else child_id,
                content=content or "",
                source=source,
                score=rrf_score,
            ))

        return docs

    # ── fusion logic ───────────────────────────────────────────

    def _rrf_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[dict],
        k: int = 60,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Two-way RRF: BM25 + Dense."""
        return self._rrf_fusion_3way(bm25_results, dense_results, [], k=k, top_k=top_k)

    def _rrf_fusion_3way(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[dict],
        image_results: list[dict],
        k: int = 60,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Three-way RRF: BM25 + Dense(text) + Dense(images)."""
        scores: dict[str, float] = {}

        for rank, (cid, _) in enumerate(bm25_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        for rank, hit in enumerate(dense_results):
            cid = hit.get("child_id", hit.get("id", ""))
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        for rank, hit in enumerate(image_results):
            cid = hit.get("child_id", hit.get("id", ""))
            if cid:
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
