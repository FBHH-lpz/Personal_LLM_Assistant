"""Retrieval node — runs multi-query hybrid search and merges results."""

from __future__ import annotations

import logging

from app.core.graph.state import RAGState
from app.core.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


async def retrieve_node(state: RAGState, retriever: HybridRetriever) -> dict:
    """Multi-query hybrid retrieval node.

    Searches with all rewritten query variants, then merges via RRF
    to produce a single ranked result list.
    """
    queries = state.get("rewrite_queries", [])
    if not queries:
        queries = [state.get("rewritten_query") or state.get("user_query", "")]

    top_k = state.get("retrieval_top_k", 20)

    logger.info("Retrieving with %d query variants (top_k=%d)", len(queries), top_k)

    # Search with each variant
    all_results: dict[str, dict] = {}  # child_id → doc
    all_rankings: list[list[str]] = []  # per-query rankings

    try:
        for q in queries[:5]:  # max 5 queries to avoid bloat
            docs = await retriever.search(q, top_k=top_k)
            ranking = []
            for d in docs:
                if d.child_id not in all_results:
                    all_results[d.child_id] = {
                        "child_id": d.child_id,
                        "parent_id": d.parent_id,
                        "content": d.content,
                        "source": d.source,
                        "score": d.score,
                    }
                ranking.append(d.child_id)
            all_rankings.append(ranking)
    except Exception:
        logger.exception("Retrieval failed")
        return {"retrieved_docs": []}

    # RRF merge across all query rankings
    merged = _rrf_merge(all_rankings, k=60)
    retrieved = [all_results[cid] for cid, _score in merged if cid in all_results]

    logger.info("Retrieved %d unique documents from %d queries",
                len(retrieved), len(queries))
    return {"retrieved_docs": retrieved}


def _rrf_merge(
    rankings: list[list[str]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]
