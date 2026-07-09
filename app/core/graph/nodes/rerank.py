"""Re-rank node — applies CrossEncoder to refine retrieval results."""

from __future__ import annotations

import asyncio
import logging

from app.core.graph.state import RAGState
from app.core.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


async def rerank_node(state: RAGState, reranker: CrossEncoderReranker) -> dict:
    """CrossEncoder re-ranking node.

    Takes the retrieved_docs from the retrieve node, scores them with
    a CrossEncoder, and returns the top-N.
    """
    docs = state.get("retrieved_docs", [])
    query = state.get("rewritten_query") or state.get("user_query", "")
    top_k = state.get("rerank_top_k", 5)

    if not docs:
        logger.warning("No documents to rerank")
        return {"reranked_docs": []}

    logger.info("Re-ranking %d documents for query: '%s'", len(docs), query[:100])

    # Extract content strings
    contents = [d["content"] for d in docs]

    try:
        scored = await reranker.rerank(query, contents, top_k=top_k)
    except Exception:
        logger.exception("Re-ranking failed, falling back to retrieval order")
        return {"reranked_docs": docs[:top_k]}

    # Map scores back to original doc dicts
    content_to_score = {text: score for text, score in scored}
    reranked = []
    for doc in docs:
        if doc["content"] in content_to_score:
            doc["score"] = float(content_to_score[doc["content"]])
            reranked.append(doc)

    # Sort by score descending, take top_k
    reranked.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    result = reranked[:top_k]

    logger.info("Re-ranked to %d documents (top score: %.3f)",
                len(result), result[0]["score"] if result else 0.0)

    return {"reranked_docs": result}
