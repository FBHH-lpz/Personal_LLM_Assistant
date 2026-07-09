"""Response generation node — builds prompt with context and calls LLM."""

from __future__ import annotations

import logging
from typing import AsyncIterator

from app.core.graph.state import RAGState
from app.core.llm.base import ChatModel

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """你是一个课程知识库问答助手。你的知识来源于已导入的课件 PDF。

规则：
1. 基于课件内容**详细充分**地回答，展开要点，包含具体例子、公式、数据，不要只给一句话
2. 如果知识库包含答案，在回答末尾用 [来源: 文件名] 标注
3. 用清晰的段落结构组织回答，必要时用列表或分点
4. 如果知识库不包含答案，如实告知用户，不要编造
5. 不要说"根据参考文档"——课件是系统内置的，不是用户上传的"""


def build_context(docs: list[dict]) -> str:
    """Build a context string from reranked documents."""
    if not docs:
        return "（知识库中未找到相关内容）"

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
            {"role": "system", "content": "你是一个课程AI助手。请简洁自然地回答用户。"},
            {"role": "user", "content": query},
        ]
    else:
        context = build_context(docs)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"知识库中的课件内容：\n{context}\n\n用户问题：{query}"},
        ]

    try:
        response = await model.achat(
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            stream=stream,
        )
    except Exception:
        logger.exception("Generation failed")
        return {"final_response": "抱歉，回答生成时出现了错误，请稍后重试。"}

    if stream:
        # Return an empty placeholder; actual content flows through events
        return {"final_response": "__STREAMING__"}

    return {"final_response": response.content}
