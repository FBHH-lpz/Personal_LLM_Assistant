"""Chat route — SSE streaming RAG responses."""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import get_db, get_graph
from app.db.models import Conversation
from app.db.sessions import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class SendMessageRequest(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    user_id: str = "default"


async def _load_history(conversation_id: str, db: AsyncSession) -> list[dict]:
    """Load conversation messages from DB."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        return []
    try:
        return json.loads(conv.messages_json)
    except (json.JSONDecodeError, TypeError):
        return []


async def _save_history(conversation_id: str, messages: list[dict], db: AsyncSession) -> None:
    """Save conversation messages to DB."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        conv = Conversation(id=conversation_id, user_id="default", title="")
        db.add(conv)

    conv.messages_json = json.dumps(messages, ensure_ascii=False)
    # Auto-title from first user message
    if not conv.title:
        for m in messages:
            if m["role"] == "user":
                conv.title = m["content"][:50]
                break
    await db.commit()


@router.post("")
async def send_message(
    req: SendMessageRequest,
    graph=Depends(get_graph),
    db_session: AsyncSession = Depends(get_db),
):
    """Send a message and receive a streaming RAG response via SSE.

    Event stream format::

        data: {"delta": "文本片段"}
        data: {"meta": {"rewritten_query": "...", "sources": [...]}}
        data: [DONE]
    """
    # Load or create conversation
    if req.conversation_id:
        history = await _load_history(req.conversation_id, db_session)
    else:
        history = []

    async def event_generator():
        # Add user message to state
        current_messages = list(history)
        current_messages.append({"role": "user", "content": req.content})

        config = {"configurable": {"thread_id": req.conversation_id or "default"}}

        try:
            full_response = ""
            # Run the graph
            result = await graph.ainvoke(
                {
                    "user_query": req.content,
                    "messages": current_messages,
                    "retrieval_top_k": 20,
                    "rerank_top_k": 5,
                },
                config=config,
            )
            final_text = result.get("final_response", "")
            rewritten = result.get("rewritten_query", "")

            # Send metadata
            if rewritten and rewritten != req.content:
                yield {
                    "event": "meta",
                    "data": json.dumps({"rewritten_query": rewritten}, ensure_ascii=False),
                }

            # Send full response in chunks (simulate streaming)
            chunk_size = 50
            for i in range(0, len(final_text), chunk_size):
                chunk = final_text[i:i+chunk_size]
                yield {
                    "event": "delta",
                    "data": json.dumps({"delta": chunk}, ensure_ascii=False),
                }

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            logger.exception("Error in chat stream")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("/{conversation_id}")
async def send_message_to_conversation(
    conversation_id: str,
    req: SendMessageRequest,
    graph=Depends(get_graph),
    db_session: AsyncSession = Depends(get_db),
):
    """Send a message to an existing conversation (streaming SSE)."""
    req.conversation_id = conversation_id
    return await send_message(req, graph, db_session)
