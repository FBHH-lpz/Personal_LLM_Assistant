#!/usr/bin/env python3
"""LLM-assisted eval dataset generation from ingested documents.

Usage:
    python eval/generate_dataset.py --sample 200 --output eval/dataset.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.llm.registry import get_cheap_chat_model
from app.core.retrieval.bm25_index import BM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_GEN_PROMPT = """阅读以下文档片段，生成 2 个不同的中文问题，
这些问题的答案必须能够从该片段中找到。
问题应该模拟真实用户可能问的方式。

文档片段：
{chunk}

返回 JSON 格式（只返回 JSON，不要其他文字）：
[
  {{"query": "问题1", "keywords": ["关键词1", "关键词2"]}},
  {{"query": "问题2", "keywords": ["关键词1", "关键词2"]}}
]"""


async def generate_queries_from_chunk(chunk_text: str, model) -> list[dict]:
    """Ask LLM to generate queries from a document chunk."""
    prompt = DATASET_GEN_PROMPT.format(chunk=chunk_text[:2000])

    try:
        result = await model.achat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        # Parse JSON from response
        content = result.content.strip()
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(content[json_start:json_end])
    except Exception as e:
        logger.warning("Failed to generate queries: %s", e)

    return []


async def main():
    parser = argparse.ArgumentParser(description="Generate eval dataset")
    parser.add_argument("--sample", type=int, default=200, help="Number of parent chunks to sample")
    parser.add_argument("--output", default="eval/dataset.jsonl", help="Output file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load BM25 index to get parent chunks
    bm25 = BM25Index()
    if not bm25.load(settings.bm25_index_path):
        logger.error("BM25 index not found. Run ingestion first: python scripts/ingest_docs.py")
        sys.exit(1)

    # Collect parent chunks
    parents = list(bm25._parents.items()) if hasattr(bm25, "_parents") else []
    if not parents:
        logger.error("No parent chunks found in BM25 index")
        sys.exit(1)

    logger.info("Found %d parent chunks in index", len(parents))

    # Sample evenly
    sample_size = min(args.sample, len(parents))
    sampled = random.sample(parents, sample_size)

    logger.info("Sampled %d chunks for dataset generation", sample_size)

    # Generate queries
    model = get_cheap_chat_model()
    all_entries: list[dict] = []

    for i, (parent_id, (content, metadata)) in enumerate(sampled):
        if not content.strip():
            continue

        queries = await generate_queries_from_chunk(content, model)

        # Find child IDs belonging to this parent
        child_ids = []
        for cid in bm25._doc_ids:
            if cid.startswith(parent_id + "_c_"):
                child_ids.append(cid)

        for q in queries:
            all_entries.append({
                "query": q["query"],
                "relevant_chunks": child_ids,
                "type": "single_turn",
                "keywords": q.get("keywords", []),
            })

        if (i + 1) % 20 == 0:
            logger.info("Progress: %d/%d chunks processed, %d queries generated",
                        i + 1, sample_size, len(all_entries))

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Dataset saved to %s (%d queries)", output_path, len(all_entries))


if __name__ == "__main__":
    asyncio.run(main())
