"""Query Rewriting node — resolves pronouns and completes intent for retrieval.

This is the key to multi-turn conversation context management.
It takes the conversation history and the user's latest message,
uses a cheap LLM to rewrite it into a standalone retrieval query.
"""

from __future__ import annotations

import logging
from typing import Union

from app.core.llm.base import ChatModel

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手。根据对话历史，将用户的当前问题改写为一个独立、完整的检索查询。

规则：
1. 所有代词（它、这个、那个、其、该等）必须替换为对话历史中提到的具体指代对象
2. 补充被省略的主语和上下文信息
3. 改写后的查询应该能够独立用于向量检索，不依赖历史对话
4. 保留用户问题的原始意图和专业术语

输出格式要求：
- 只输出一行改写后的查询文本，不要添加任何解释或标记
- 如果用户只是寒暄/打招呼/感谢，输出: EMPTY"""


async def rewrite_query(
    user_query: str,
    history: list[dict[str, str]],
    model: ChatModel,
    max_history_turns: int = 6,
) -> tuple[str, bool]:
    """Rewrite user query for standalone retrieval.

    Args:
        user_query: The user's raw latest message.
        history: Conversation history as [{"role": "user/assistant", "content": "..."}, ...].
        model: A cheap chat model for the rewriting task.
        max_history_turns: How many recent turns to include as context.

    Returns:
        (rewritten_query, needs_retrieval)
        - rewritten_query: The standalone query for retrieval, or "EMPTY" if no retrieval needed.
        - needs_retrieval: False if rewritten_query == "EMPTY" (greeting/chitchat).
    """
    # Truncate history to recent N turns
    recent_history = history[-(max_history_turns * 2):] if history else []

    # Build messages for the rewrite model
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
    ]

    # Include history as context
    for msg in recent_history:
        messages.append(msg)

    # Add current query
    messages.append({"role": "user", "content": user_query})

    try:
        response = await model.achat(
            messages=messages,
            temperature=0.0,       # deterministic for rewriting
            max_tokens=200,        # rewritten query should be short
            stream=False,
        )
    except Exception:
        logger.exception("Query rewriting failed, using original query")
        return user_query, True

    rewritten = response.content.strip()

    # Remove quotes that LLMs sometimes add
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1]
    if rewritten.startswith("'") and rewritten.endswith("'"):
        rewritten = rewritten[1:-1]

    # Check if no retrieval needed
    if rewritten.upper() == "EMPTY" or rewritten == "EMPTY":
        logger.info("Query rewrite: '%s' → EMPTY (chitchat)", user_query[:50])
        return rewritten, False

    logger.info("Query rewrite: '%s' → '%s'", user_query[:80], rewritten[:80])
    return rewritten, True
