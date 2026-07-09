"""LangGraph state definition for the RAG pipeline."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """State that flows through the RAG graph nodes.

    Fields are updated by each node; LangGraph merges the return dict into the full state.
    """
    # ── Input ──────────────────────────────────────────────────
    messages: Annotated[list[dict[str, str]], add_messages]
    """Full conversation history. Managed by LangGraph checkpointing."""

    user_query: str
    """The user's raw latest message."""

    # ── Rewrite ────────────────────────────────────────────────
    rewritten_query: str
    """Primary query after pronoun resolution and intent completion."""

    rewrite_queries: list[str]
    """All rewritten query variants for multi-query retrieval."""

    needs_retrieval: bool
    """Whether this message requires document retrieval."""

    # ── Retrieval ──────────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]
    """Documents retrieved by hybrid search (child-level metadata)."""

    # ── Rerank ─────────────────────────────────────────────────
    reranked_docs: list[dict[str, Any]]
    """Top-N documents after CrossEncoder re-ranking."""

    # ── Generation ─────────────────────────────────────────────
    final_response: str
    """The assistant's final response text."""


def initial_state(user_query: str, messages: list[dict[str, str]] | None = None) -> RAGState:
    """Create a fresh state for each turn."""
    return RAGState(
        messages=messages or [],
        user_query=user_query,
        rewritten_query="",
        rewrite_queries=[],
        needs_retrieval=True,
        retrieved_docs=[],
        reranked_docs=[],
        final_response="",
    )
