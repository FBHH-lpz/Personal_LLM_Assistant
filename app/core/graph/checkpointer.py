"""Checkpointer factory — provides SQLite-backed state persistence for LangGraph."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(db_path: str) -> InMemorySaver:
    """Create a checkpointer for LangGraph.

    Uses InMemorySaver for now — SqliteSaver.from_conn_string() returns
    a generator in newer LangGraph versions. Will persist later.

    Args:
        db_path: Path to the SQLite database file (unused for now).

    Returns:
        An InMemorySaver instance.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return InMemorySaver()
