"""SQLAlchemy ORM models for conversation management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(128), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", lazy="selectin")
    documents: Mapped[list["Document"]] = relationship(back_populates="user", lazy="selectin")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(256), default="新对话")
    messages_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of {role, content}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    user: Mapped["User"] = relationship(back_populates="conversations")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="processing")  # processing / ready / error
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="documents")
