"""CrossEncoder re-ranker for fine-grained semantic scoring.

Uses BAAI/bge-reranker-v2-m3 (multilingual, strong on Chinese/English).
Takes the top-N candidates from hybrid retrieval and re-ranks them.
"""

from __future__ import annotations

import asyncio
import logging
from functools import cache
from typing import Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank retrieved documents using a CrossEncoder model.

    Usage::

        reranker = CrossEncoderReranker()
        await reranker.ensure_loaded()
        scored = await reranker.rerank("query text", ["doc1", "doc2", ...], top_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model: Optional[object] = None  # sentence_transformers.CrossEncoder

    async def ensure_loaded(self) -> None:
        """Load the model if not already loaded (runs in executor to avoid blocking)."""
        if self._model is not None:
            return

        def _load():
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder: %s on %s", self.model_name, self.device)
            return CrossEncoder(self.model_name, device=self.device)

        self._model = await asyncio.get_event_loop().run_in_executor(None, _load)
        logger.info("CrossEncoder loaded: %s", self.model_name)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Score and re-rank documents against the query.

        Args:
            query: The search query.
            documents: Candidate document texts.
            top_k: Number of top results to return.

        Returns:
            List of (document_text, score) sorted by score descending.
        """
        if not documents:
            return []

        await self.ensure_loaded()

        # Build (query, doc) pairs
        pairs = [(query, doc) for doc in documents]

        def _predict():
            return self._model.predict(pairs, show_progress_bar=False)

        scores = await asyncio.get_event_loop().run_in_executor(None, _predict)

        # Pair and sort
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    async def rerank_retrieved(
        self,
        query: str,
        docs: list,  # list of RetrievedDoc
        top_k: int = 5,
    ) -> list:
        """Convenience: rerank RetrievedDoc objects and return re-ranked list."""
        if not docs:
            return []

        contents = [d.content for d in docs]
        scored = await self.rerank(query, contents, top_k=top_k)

        # Map scores back to RetrievedDoc objects
        content_to_score = {text: score for text, score in scored}
        result = []
        for doc in docs:
            if doc.content in content_to_score:
                doc.score = content_to_score[doc.content]
                result.append(doc)

        result.sort(key=lambda d: d.score, reverse=True)
        return result[:top_k]
