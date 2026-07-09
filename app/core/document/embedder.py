"""Batch embedding generation via cloud API."""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from app.core.llm.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Maximum batch size per API call (provider-dependent; Tongyi allows up to ~100)
EMBED_BATCH_SIZE = 10
# Max concurrent API calls
MAX_CONCURRENT = 5


async def embed_batch(
    texts: Sequence[str],
    provider: EmbeddingProvider,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts in batches with concurrency control.

    Args:
        texts: List of text strings to embed.
        provider: An EmbeddingProvider instance.
        batch_size: How many texts per API call.

    Returns:
        List of embedding vectors, one per input text.
    """
    if not texts:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _embed_one_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            resp = await provider.aembed(batch)
            return resp.embeddings

    # Split into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    logger.info("Embedding %d texts in %d batches (batch_size=%d)", len(texts), len(batches), batch_size)

    # Run all batches concurrently
    results = await asyncio.gather(*[_embed_one_batch(b) for b in batches])

    # Flatten
    all_embeddings: list[list[float]] = []
    for batch_result in results:
        all_embeddings.extend(batch_result)

    return all_embeddings
