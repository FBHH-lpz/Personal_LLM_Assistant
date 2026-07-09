"""Retrieval node — calls HybridRetriever and populates state."""

from __future__ import annotations

import logging

from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.graph.state import RAGState

logger = logging.getLogger(__name__)


async def retrieve_node(state: RAGState, retriever: HybridRetriever) -> dict:
    """Hybrid retrieval node for the LangGraph pipeline.

    Uses the rewritten_query from the rewrite node to search.
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    top_k = state.get("retrieval_top_k", 20)

    logger.info("Retrieving for query: '%s' (top_k=%d)", query[:100], top_k)

    try:
        docs = await retriever.search(query, top_k=top_k)
    except Exception:
        logger.exception("Retrieval failed")
        return {"retrieved_docs": []}

    # Serialize RetrievedDoc objects for the state
    retrieved = [
        {
            "child_id": d.child_id,
            "parent_id": d.parent_id,
            "content": d.content,
            "source": d.source,
            "score": d.score,
        }
        for d in docs
    ]

    logger.info("Retrieved %d documents", len(retrieved))
    return {"retrieved_docs": retrieved}
