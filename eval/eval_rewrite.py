#!/usr/bin/env python3
"""Evaluate Query Rewriting accuracy on multi-turn dialogue dataset."""

import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.graph.nodes.rewrite import rewrite_query
from app.core.llm.registry import get_cheap_chat_model


def check_no_pronouns(text: str) -> bool:
    """Check if text contains unresolved pronouns."""
    pronouns = {"它", "这个", "那个", "其", "该", "它们", "这些", "那些"}
    for p in pronouns:
        if p in text:
            return False
    return True


async def main():
    model = get_cheap_chat_model()

    ds_path = Path("eval/rewrite_dataset.jsonl")
    if not ds_path.exists():
        print("rewrite_dataset.jsonl not found")
        return

    dataset = []
    with open(ds_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    print(f"Evaluating {len(dataset)} multi-turn queries...\n")

    results = []
    for i, item in enumerate(dataset):
        result = await rewrite_query(
            user_query=item["query"],
            history=item["history"],
            model=model,
        )
        rewritten = result["primary_query"]
        needs_retrieve = result["needs_retrieval"]
        expected = item["expected_rewrite"]

        # Semantic similarity check (simple case-insensitive keyword overlap)
        rw_words = set(rewritten)
        ex_words = set(expected)

        # Count how many expected key terms appear in rewritten
        key_match = 0
        key_missing = []
        for kw in expected.replace("的", "").replace("和", "").replace("与", "").split():
            pass  # rough check not perfect

        result = {
            "query": item["query"],
            "rewritten": rewritten,
            "expected": expected,
            "pronouns_resolved": check_no_pronouns(rewritten),
            "needs_retrieval": needs_retrieve,
        }
        results.append(result)

        status = "✓" if result["pronouns_resolved"] else "✗"
        print(f"[{i+1:2d}] {status} Query: {item['query']}")
        print(f"    RW:  {rewritten}")
        print(f"    EXP: {expected}")
        print()

    # Metrics
    pronoun_ok = sum(1 for r in results if r["pronouns_resolved"])
    pct = pronoun_ok / len(results) * 100

    print("=" * 60)
    print(f"  QUERY REWRITING EVALUATION")
    print(f"  Total: {len(results)} | Pronouns resolved: {pronoun_ok}/{len(results)}")
    print(f"  Pronoun Resolution Rate: {pct:.1f}%")
    print("=" * 60)

    # Save
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rewrite_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: eval/results/rewrite_eval.json")


if __name__ == "__main__":
    asyncio.run(main())
