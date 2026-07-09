"""Conversation CRUD routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.db.models import Conversation, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationResponse]


async def _conv_to_response(conv: Conversation) -> ConversationResponse:
    messages = []
    try:
        messages = json.loads(conv.messages_json)
    except (json.JSONDecodeError, TypeError):
        pass
    return ConversationResponse(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title or "未命名对话",
        message_count=len(messages),
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
    )


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for a user."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    convs = result.scalars().all()
    items = [await _conv_to_response(c) for c in convs]
    return ConversationListResponse(conversations=items)


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single conversation with messages."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    try:
        messages = json.loads(conv.messages_json)
    except (json.JSONDecodeError, TypeError):
        pass

    return {
        "id": conv.id,
        "user_id": conv.user_id,
        "title": conv.title,
        "messages": messages,
        "created_at": conv.created_at.isoformat() if conv.created_at else "",
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await db.delete(conv)
    await db.commit()
    return {"status": "deleted", "id": conversation_id}
