"""Chat route — true SSE streaming with retrieval + LLM streaming."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_chat, get_db, get_graph
from app.db.models import Conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class SendMessageRequest(BaseModel):
    content: str
    conversation_id: Optional[str] = None
    user_id: str = "default"


async def _load_history(conversation_id: str, db: AsyncSession) -> list[dict]:
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


async def _save_history(
    conversation_id: str, messages: list[dict], db: AsyncSession,
) -> None:
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if conv is None:
        conv = Conversation(id=conversation_id, user_id="default", title="")
        db.add(conv)
    conv.messages_json = json.dumps(messages, ensure_ascii=False)
    if not conv.title or conv.title == "新对话":
        for m in messages:
            if m["role"] == "user":
                conv.title = m["content"][:80]
                break
    await db.commit()


@router.post("")
async def send_message(
    req: SendMessageRequest,
    graph=Depends(get_graph),
    chat_model=Depends(get_chat),
    db_session: AsyncSession = Depends(get_db),
):
    """Send a message with true SSE streaming.

    Retrieval runs first, then LLM tokens are streamed in real-time.
    """
    if not req.conversation_id:
        req.conversation_id = uuid.uuid4().hex[:12]

    history = await _load_history(req.conversation_id, db_session)
    current_messages = list(history)
    current_messages.append({"role": "user", "content": req.content})

    config = {"configurable": {"thread_id": req.conversation_id or "default"}}

    async def event_generator():
        nonlocal current_messages
        full_response = ""
        rewritten = ""

        try:
            # ── Step 1: Run graph (rewrite → retrieve → rerank) ──
            # We pass a special flag to skip the respond node
            result = await graph.ainvoke(
                {
                    "user_query": req.content,
                    "messages": current_messages,
                    "retrieval_top_k": 20,
                    "rerank_top_k": 5,
                },
                config=config,
            )
            rewritten = result.get("rewritten_query", "")
            docs = result.get("reranked_docs", [])

            # ── Send meta ──────────────────────────────────────────
            meta = {"conversation_id": req.conversation_id}
            if rewritten and rewritten != req.content:
                meta["rewritten_query"] = rewritten
            if docs:
                meta["sources"] = list(set(d.get("source", "") for d in docs[:5]))
            yield {"event": "meta", "data": json.dumps(meta, ensure_ascii=False)}

            # ── Step 2: Build prompt ───────────────────────────────
            from app.core.graph.nodes.respond import RAG_SYSTEM_PROMPT, build_context

            needs_retrieval = result.get("needs_retrieval", True) if "needs_retrieval" in result else True
            query = rewritten or req.content

            if not needs_retrieval or not docs:
                messages = [
                    {"role": "system", "content": "你是一个课程AI助手。请简洁自然地回答用户。"},
                    {"role": "user", "content": query},
                ]
            else:
                context = build_context(docs)
                messages = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"知识库中的课件内容：\n{context}\n\n用户问题：{query}"},
                ]

            # ── Step 3: Stream LLM tokens in real-time ─────────────
            async for chunk in chat_model.achat(
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                stream=True,
            ):
                if chunk.delta_content:
                    full_response += chunk.delta_content
                    yield {
                        "event": "delta",
                        "data": json.dumps({"delta": chunk.delta_content}, ensure_ascii=False),
                    }

            # ── Step 4: Save history ──────────────────────────────
            if full_response:
                current_messages.append({"role": "assistant", "content": full_response})
                if req.conversation_id:
                    try:
                        await _save_history(req.conversation_id, current_messages, db_session)
                    except Exception:
                        logger.exception("Failed to save history")

            yield {"event": "done", "data": "[DONE]"}

        except Exception as e:
            logger.exception("Chat error")
            yield {"event": "error", "data": json.dumps({"error": str(e)}, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


@router.post("/{conversation_id}")
async def send_message_to_conversation(
    conversation_id: str,
    req: SendMessageRequest,
    graph=Depends(get_graph),
    chat_model=Depends(get_chat),
    db_session: AsyncSession = Depends(get_db),
):
    """Send a message to an existing conversation with SSE streaming."""
    req.conversation_id = conversation_id
    return await send_message(req, graph, chat_model, db_session)
