"""Tests for LangGraph nodes and graph construction."""

from __future__ import annotations

import pytest


class TestRewriteNode:
    """Tests for query rewriting."""

    @pytest.mark.asyncio
    async def test_rewrite_without_history(self):
        """Query with no history should remain mostly intact."""
        from app.core.graph.nodes.rewrite import rewrite_query

        # We skip this test if no LLM configured (unit test isolation)
        pytest.skip("Requires LLM API — integration test only")

    def test_rewrite_prompt_format(self):
        """The rewrite system prompt should contain key instructions."""
        from app.core.graph.nodes.rewrite import REWRITE_SYSTEM_PROMPT

        assert "查询改写" in REWRITE_SYSTEM_PROMPT
        assert "代词" in REWRITE_SYSTEM_PROMPT
        assert "EMPTY" in REWRITE_SYSTEM_PROMPT


class TestRespondNode:
    """Tests for response generation."""

    def test_build_context(self):
        from app.core.graph.nodes.respond import build_context

        docs = [
            {"source": "test.pdf", "content": "This is document content."},
            {"source": "test2.pdf", "content": "More content here."},
        ]
        ctx = build_context(docs)

        assert "test.pdf" in ctx
        assert "test2.pdf" in ctx
        assert "This is document content" in ctx

    def test_build_context_empty(self):
        from app.core.graph.nodes.respond import build_context

        ctx = build_context([])
        assert "未找到" in ctx


class TestRAGState:
    """Tests for state creation."""

    def test_initial_state(self):
        from app.core.graph.state import initial_state

        state = initial_state("test query")
        assert state["user_query"] == "test query"
        assert state["messages"] == []
        assert state["needs_retrieval"] is True
        assert state["rewritten_query"] == ""
        assert state["retrieved_docs"] == []
        assert state["reranked_docs"] == []
        assert state["final_response"] == ""


class TestGraphConstruction:
    """Tests for graph builder."""

    def test_graph_builds(self):
        """Graph should compile without error (with mocked dependencies)."""
        import pytest
        pytest.skip("Requires full dependency graph — integration test only")
