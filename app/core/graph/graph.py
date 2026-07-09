"""LangGraph builder — assembles the RAG pipeline graph."""

from __future__ import annotations

import logging
from functools import partial

from langgraph.graph import END, StateGraph

from app.config import settings as global_settings
from app.core.graph.checkpointer import create_sqlite_checkpointer
from app.core.graph.nodes.respond import respond_node
from app.core.graph.nodes.retrieve import retrieve_node
from app.core.graph.nodes.rerank import rerank_node
from app.core.graph.nodes.rewrite import rewrite_query
from app.core.graph.state import RAGState
from app.core.llm.base import ChatModel
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def _route_after_rewrite(state: RAGState) -> str:
    """Conditional routing: skip retrieval for chitchat."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "respond"


def build_rag_graph(
    chat_model: ChatModel,
    cheap_model: ChatModel,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> StateGraph:
    """Build and compile the RAG LangGraph.

    Graph structure::

        rewrite ──(needs_retrieval?)──→ retrieve ──→ rerank ──→ respond
            │                                                      ↑
            └──────────(chitchat)──────────────────────────────────┘

    Args:
        chat_model: Primary LLM for response generation.
        cheap_model: Cheap LLM for query rewriting.
        retriever: Hybrid retriever (BM25 + Dense).
        reranker: CrossEncoder re-ranker.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    g = StateGraph(RAGState)

    # ── Define nodes ───────────────────────────────────────

    async def _rewrite(state: RAGState) -> dict:
        rewritten, needs_retrieval = await rewrite_query(
            user_query=state["user_query"],
            history=state.get("messages", []),
            model=cheap_model,
            max_history_turns=global_settings.rewrite_history_turns,
        )
        return {
            "rewritten_query": rewritten,
            "needs_retrieval": needs_retrieval,
        }

    async def _retrieve(state: RAGState) -> dict:
        return await retrieve_node(state, retriever)

    async def _rerank(state: RAGState) -> dict:
        return await rerank_node(state, reranker)

    async def _respond(state: RAGState) -> dict:
        return await respond_node(state, chat_model, stream=False)

    g.add_node("rewrite", _rewrite)
    g.add_node("retrieve", _retrieve)
    g.add_node("rerank", _rerank)
    g.add_node("respond", _respond)

    # ── Define edges ───────────────────────────────────────

    g.set_entry_point("rewrite")

    g.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {
            "retrieve": "retrieve",
            "respond": "respond",
        },
    )

    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "respond")
    g.add_edge("respond", END)

    # ── Compile with checkpointing ─────────────────────────

    checkpointer = create_sqlite_checkpointer(global_settings.sqlite_db_path)
    compiled = g.compile(checkpointer=checkpointer)

    logger.info("RAG graph compiled successfully")
    return compiled
