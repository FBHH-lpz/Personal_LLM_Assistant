"""Database session manager."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = logging.getLogger(__name__)

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Initialize database engine and create tables."""
    global _engine, _sessionmaker

    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Use aiosqlite for async SQLite
    db_url = f"sqlite+aiosqlite:///{settings.sqlite_db_path}"

    _engine = create_async_engine(db_url, echo=settings.debug)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)

    # Create tables
    from app.db.models import Base
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized at %s", settings.sqlite_db_path)


async def get_session() -> AsyncSession:
    """Get an async database session."""
    if _sessionmaker is None:
        await init_db()
    async with _sessionmaker() as session:
        yield session


async def close_db() -> None:
    """Close database connections."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database connections closed")
