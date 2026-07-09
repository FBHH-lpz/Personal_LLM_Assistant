"""Checkpointer factory — provides SQLite-backed state persistence for LangGraph."""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(db_path: str) -> SqliteSaver:
    """Create a SQLite-backed LangGraph checkpointer.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A SqliteSaver instance ready to be passed to graph.compile().
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # SqliteSaver takes a connection; we use from_conn_string for path-based
    return SqliteSaver.from_conn_string(db_path)
