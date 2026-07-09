"""Response generation node — builds prompt with context and calls LLM."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from app.core.graph.state import RAGState
from app.core.llm.base import ChatModel

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是一个知识库问答助手。请根据提供的参考文档回答用户的问题。

规则：
1. 如果参考文档包含答案，基于文档内容回答，并在回答末尾用 [来源: 文件名] 标注
2. 如果参考文档不包含答案，如实告知用户，不要编造
3. 回答要简洁、准确、有条理
4. 如果用户问的是对话历史中讨论过的内容，结合历史和文档一起回答"""


def build_context(docs: list[dict]) -> str:
    """Build a context string from reranked documents."""
    if not docs:
        return "（未找到相关参考文档）"

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        content = doc.get("content", "")
        parts.append(f"[文档{i}] 来源: {source}\n{content}")

    return "\n\n---\n\n".join(parts)


async def respond_node(
    state: RAGState,
    model: ChatModel,
    stream: bool = False,
) -> dict:
    """Generate the final response using the LLM.

    Args:
        state: Current RAG state.
        model: The primary LLM for generation.
        stream: Whether the caller handles streaming externally.
            When True, only stores a placeholder and caller reads from events.

    Returns:
        dict with 'final_response' key.
    """
    query = state.get("rewritten_query") or state.get("user_query", "")
    docs = state.get("reranked_docs", [])
    needs_retrieval = state.get("needs_retrieval", True)

    if not needs_retrieval or not docs:
        # Greeting or chitchat — no context needed
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "你是一个友好的AI助手。请简洁自然地回答用户。"},
            {"role": "user", "content": query},
        ]
    else:
        context = build_context(docs)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"参考文档：\n{context}\n\n用户问题：{query}"},
        ]

    try:
        response = await model.achat(
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            stream=stream,
        )
    except Exception:
        logger.exception("Generation failed")
        return {"final_response": "抱歉，回答生成时出现了错误，请稍后重试。"}

    if stream:
        # Return an empty placeholder; actual content flows through events
        return {"final_response": "__STREAMING__"}

    return {"final_response": response.content}
