"""Tongyi Qwen-VL vision-language model client for chart/image analysis.

Features:
- Structured JSON output (type, title, data_points, key_insights)
- Concurrent VLM calls via asyncio.Semaphore
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path

import aiohttp

from app.config import settings

logger = logging.getLogger(__name__)

# Max concurrent VLM API calls
VLM_CONCURRENCY = 5
_vlm_semaphore = asyncio.Semaphore(VLM_CONCURRENCY)

IMAGE_STRUCTURED_PROMPT = """分析这张图片，返回结构化 JSON。不要写解释，只返回 JSON：

如果是**图表**（柱状图、折线图、饼图、散点图等）：
{
  "type": "chart",
  "chart_type": "柱状图/折线图/饼图等",
  "title": "图表标题",
  "axes": {"x": "横轴标签及含义", "y": "纵轴标签及含义"},
  "data_points": [
    {"label": "数据项名称", "value": "数值或趋势描述"},
    ...
  ],
  "key_insights": ["核心发现1", "核心发现2"]
}

如果是**表格**：
{
  "type": "table",
  "title": "表格标题",
  "headers": ["列名1", "列名2", ...],
  "rows": [["值", "值", ...], ...],
  "key_insights": ["核心发现1"]
}

如果是**流程图/示意图**：
{
  "type": "diagram",
  "title": "图示标题",
  "description": "图中展示的概念和流程",
  "steps": ["步骤1", "步骤2", ...],
  "key_insights": ["核心概念"]
}

如果是**公式**：
{
  "type": "formula",
  "content": "公式转录（LaTeX格式）",
  "variables": [{"symbol": "符号", "meaning": "含义"}, ...],
  "explanation": "公式解释"
}

如果是普通插图/封面/装饰性图片：
{
  "type": "other",
  "description": "简短描述"
}
"""


async def describe_image_structured(
    image_path: str | Path,
) -> dict | None:
    """Send an image to Qwen-VL and get a structured JSON description.

    Uses concurrent VLM calls with semaphore control.

    Returns parsed JSON dict, or None on failure.
    """
    async with _vlm_semaphore:
        return await _call_vlm(image_path)


async def _call_vlm(image_path: str | Path) -> dict | None:
    """Internal: single VLM API call."""
    image_path = Path(image_path)

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.suffix.lower().lstrip(".")
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    headers = {
        "Authorization": f"Bearer {settings.tongyi_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "qwen-vl-plus",
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"image": f"data:{mime_type};base64,{image_data}"},
                    {"text": IMAGE_STRUCTURED_PROMPT},
                ],
            }]
        },
        "parameters": {"max_tokens": 1000},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("VLM error %d: %s", resp.status, text[:200])
                    return None

                data = await resp.json()
                output = data.get("output", {})
                choices = output.get("choices", [])
                if not choices:
                    return None

                content = choices[0].get("message", {}).get("content", [])
                text = ""
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text += item["text"]
                    elif isinstance(item, str):
                        text += item

                # Parse JSON from response
                text = text.strip()
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    return json.loads(text[json_start:json_end])

                logger.warning("VLM returned non-JSON: %s", text[:100])
                return {"type": "other", "description": text}

    except json.JSONDecodeError:
        logger.warning("VLM JSON parse failed")
        return {"type": "other", "description": text if 'text' in dir() else ""}
    except Exception:
        logger.exception("VLM call failed")
        return None


async def describe_images_concurrent(
    image_paths: list[Path | str],
) -> list[dict | None]:
    """Describe multiple images concurrently.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of parsed JSON dicts (or None for failed calls), same order.
    """
    tasks = [describe_image_structured(p) for p in image_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if isinstance(r, dict) else None for r in results]


def structured_desc_to_text(desc: dict | None, source: str, page_num: int) -> str:
    """Convert a structured image description dict to searchable text chunk."""
    if not desc:
        return ""

    t = desc.get("type", "other")
    parts = [f"[{source} 第{page_num}页 - {t}]"]

    if t == "chart":
        parts.append(f"图表类型: {desc.get('chart_type', '')}")
        parts.append(f"标题: {desc.get('title', '')}")
        if desc.get("axes"):
            parts.append(f"横轴: {desc['axes'].get('x', '')}; 纵轴: {desc['axes'].get('y', '')}")
        for dp in desc.get("data_points", []):
            parts.append(f"  {dp.get('label', '')}: {dp.get('value', '')}")
        for ins in desc.get("key_insights", []):
            parts.append(f"核心发现: {ins}")

    elif t == "table":
        parts.append(f"表格: {desc.get('title', '')}")
        if desc.get("headers"):
            parts.append(f"列: {', '.join(desc['headers'])}")
        for row in desc.get("rows", []):
            parts.append(f"  {', '.join(str(c) for c in row)}")
        for ins in desc.get("key_insights", []):
            parts.append(f"核心发现: {ins}")

    elif t == "diagram":
        parts.append(f"示意图: {desc.get('title', '')}")
        parts.append(desc.get("description", ""))
        for s in desc.get("steps", []):
            parts.append(f"步骤: {s}")

    elif t == "formula":
        parts.append(f"公式: {desc.get('content', '')}")
        for v in desc.get("variables", []):
            parts.append(f"  {v.get('symbol', '')}: {v.get('meaning', '')}")
        parts.append(f"解释: {desc.get('explanation', '')}")

    else:
        parts.append(desc.get("description", ""))

    return "\n".join(parts)
