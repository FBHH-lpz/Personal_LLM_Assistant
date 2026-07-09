"""Query Rewriting node — advanced multi-strategy query rewriting.

Strategies:
1. Pronoun resolution + intent completion (core)
2. Multi-Query: generate 2-3 query variants for better recall
3. Query expansion: include synonyms and related terms
4. Sub-question decomposition: break complex questions apart
5. Chitchat detection: skip retrieval for greetings
"""

from __future__ import annotations

import json
import logging
from typing import Union

from app.core.llm.base import ChatModel

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手。根据对话历史，将用户的当前问题改写为用于向量检索的查询。

## 规则：
1. **指代消解**：所有代词（它、这个、那个、其、该等）必须替换为对话历史中的具体对象
2. **省略补全**：补充被省略的主语和上下文信息
3. **多角度改写**：生成 2-3 个语义等价但措辞不同的查询变体，涵盖问题的不同表述方式
4. **查询扩展**：为每个变体补充关键同义词、缩写全称、相关术语
5. **子问题分解**：如果问题是复合问题（包含多个子问题），拆分为独立的子查询

## 输出格式（严格 JSON）：

如果用户只是寒暄/打招呼/感谢，返回：
{"queries": [], "needs_retrieval": false}

否则返回：
{
  "queries": [
    "改写后的主要查询（含指代消解和省略补全）",
    "不同措辞的变体查询",
    "关键词扩展版本（包含同义词、缩写全称等）"
  ],
  "needs_retrieval": true,
  "sub_questions": []  // 如果存在子问题，填入拆分后的子查询；否则为空
}

## 示例：
历史：用户"什么是Apriori算法" / 助手"Apriori是关联规则挖掘的经典算法..."
当前问题：它有什么优缺点？

输出：
{
  "queries": [
    "Apriori算法的优缺点",
    "Apriori算法优势与劣势",
    "关联规则挖掘 Apriori 优点 缺点 局限性"
  ],
  "needs_retrieval": true,
  "sub_questions": []
}

历史：用户"SVM的原理是什么" / 助手"SVM是通过寻找最优超平面..."
当前问题：它和决策树比怎么样？各自适用什么场景？

输出：
{
  "queries": [
    "支持向量机SVM与决策树对比",
    "SVM vs Decision Tree 适用场景"
  ],
  "needs_retrieval": true,
  "sub_questions": ["SVM和决策树的区别", "SVM的适用场景", "决策树的适用场景"]
}
"""


async def rewrite_query(
    user_query: str,
    history: list[dict[str, str]],
    model: ChatModel,
    max_history_turns: int = 6,
) -> dict:
    """Rewrite user query for standalone retrieval with multi-strategy enhancement.

    Args:
        user_query: The user's raw latest message.
        history: Conversation history.
        model: A cheap chat model for the rewriting task.
        max_history_turns: How many recent turns to include as context.

    Returns:
        {
            "queries": list[str],        # rewritten queries for retrieval
            "primary_query": str,        # best single query (backward compat)
            "needs_retrieval": bool,
            "sub_questions": list[str],  # decomposed sub-questions
        }
    """
    # Truncate history
    recent_history = history[-(max_history_turns * 2):] if history else []

    # Build messages
    messages: list[dict[str, str]] = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
    ]
    for msg in recent_history:
        if isinstance(msg, dict):
            messages.append(msg)
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            messages.append({"role": msg.role, "content": msg.content})
        elif hasattr(msg, "type"):
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            messages.append({"role": role, "content": str(msg.content)})

    messages.append({"role": "user", "content": user_query})

    # Default fallback
    fallback = {
        "queries": [user_query],
        "primary_query": user_query,
        "needs_retrieval": True,
        "sub_questions": [],
    }

    try:
        response = await model.achat(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stream=False,
        )
    except Exception:
        logger.exception("Query rewriting failed, using original query")
        return fallback

    raw = response.content.strip()

    # Parse JSON
    try:
        # Find JSON block if wrapped
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(raw[json_start:json_end])
        else:
            # Fall back to treating entire response as single query
            return {**fallback, "queries": [raw], "primary_query": raw}
    except json.JSONDecodeError:
        logger.warning("Rewrite output not valid JSON: '%s'", raw[:100])
        return {**fallback, "queries": [raw], "primary_query": raw}

    # Normalize
    queries = parsed.get("queries", [])
    needs_retrieval = parsed.get("needs_retrieval", True)
    sub_questions = parsed.get("sub_questions", [])

    if not queries and needs_retrieval:
        queries = [user_query]

    # Add sub-questions to query list
    all_queries = queries + sub_questions

    primary = queries[0] if queries else user_query

    logger.info(
        "Query rewrite: '%s' → %d queries%s",
        user_query[:60],
        len(all_queries),
        f" (+{len(sub_questions)} sub)" if sub_questions else "",
    )

    return {
        "queries": all_queries,
        "primary_query": primary,
        "needs_retrieval": needs_retrieval,
        "sub_questions": sub_questions,
    }
