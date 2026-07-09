"""LLM-as-Judge evaluation for RAG answer quality.

Evaluates faithfulness (no hallucination) and answer relevance.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """你是一个评估助手。请逐句检查以下回答是否能够从给定的参考文档中找到支撑。

参考文档：
{context}

回答：
{response}

对回答的每一句话，判断它是：
- FULLY_SUPPORTED: 完全有文档支撑
- PARTIALLY_SUPPORTED: 部分有支撑
- UNSUPPORTED: 没有支撑（幻觉）

返回 JSON 格式：
{{
  "checks": [
    {{"statement": "...", "verdict": "FULLY_SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED", "reason": "..."}}
  ],
  "overall_faithfulness": 0.0
}}

overall_faithfulness = (FULLY_SUPPORTED数 + 0.5 * PARTIALLY_SUPPORTED数) / 总语句数"""


@dataclass
class FaithfulnessResult:
    score: float
    checks: list[dict]
    raw_response: str = ""


async def evaluate_faithfulness(
    response: str,
    context: str,
    model,
) -> FaithfulnessResult:
    """Evaluate whether the response is grounded in the provided context.

    Args:
        response: The assistant's answer.
        context: The retrieved documents used as context.
        model: A chat model for the evaluation.

    Returns:
        FaithfulnessResult with score and per-statement checks.
    """
    prompt = FAITHFULNESS_PROMPT.format(context=context[:8000], response=response[:4000])

    try:
        result = await model.achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            stream=False,
        )
        raw = result.content

        # Try to find JSON in the response
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(raw[json_start:json_end])
            return FaithfulnessResult(
                score=parsed.get("overall_faithfulness", 0.0),
                checks=parsed.get("checks", []),
                raw_response=raw,
            )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Faithfulness evaluation failed: %s", e)

    return FaithfulnessResult(score=0.0, checks=[])


ANSWER_RELEVANCE_PROMPT = """评估以下回答是否准确回应了用户的问题。

用户问题：{query}
回答：{response}

请打分 0-10：
- 10: 完全切题，精准回答了问题
- 7-9: 基本切题，主要内容正确
- 4-6: 部分相关，但有偏移
- 1-3: 基本不相关
- 0: 完全不相关

返回 JSON：{{"score": N, "reason": "一句话解释"}}"""


async def evaluate_answer_relevance(query: str, response: str, model) -> dict:
    """Evaluate if the answer addresses the user's question."""
    prompt = ANSWER_RELEVANCE_PROMPT.format(query=query, response=response[:2000])

    try:
        result = await model.achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            stream=False,
        )
        # Parse JSON
        json_start = result.content.find("{")
        json_end = result.content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(result.content[json_start:json_end])
    except Exception:
        pass

    return {"score": 0, "reason": "evaluation failed"}
