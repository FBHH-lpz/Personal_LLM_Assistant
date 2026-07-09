#!/usr/bin/env python3
"""Minimal eval: pure BM25 baseline. No Milvus, no CrossEncoder, no embedding API."""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.retrieval.bm25_index import BM25Index
from app.config import settings
from eval.metrics import hit_rate, mrr, avg_recall, avg_ndcg

# Load BM25 index
bm25 = BM25Index()
bm25.load(settings.bm25_index_path)
print(f"BM25: {len(bm25)} docs, {len(bm25._parents)} parents")

# Load dataset
ds_path = Path("eval/dataset.jsonl")
queries = []
with open(ds_path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            queries.append(json.loads(line))
print(f"Dataset: {len(queries)} queries")

# Run BM25 baseline (k=5)
ret_ids, rel_sets = [], []
for i, item in enumerate(queries):
    query = item["query"]
    relevant = set(item.get("relevant_chunks", []))
    results = bm25.search(query, top_k=10)
    ret_ids.append([r[0] for r in results])
    rel_sets.append(relevant)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(queries)}...")

metrics = {
    "hit_at_1": hit_rate(rel_sets, ret_ids, k=1),
    "hit_at_3": hit_rate(rel_sets, ret_ids, k=3),
    "hit_at_5": hit_rate(rel_sets, ret_ids, k=5),
    "hit_at_10": hit_rate(rel_sets, ret_ids, k=10),
    "mrr": mrr(rel_sets, ret_ids),
    "recall_at_5": avg_recall(rel_sets, ret_ids, k=5),
    "ndcg_at_5": avg_ndcg(rel_sets, ret_ids, k=5),
}

print()
print("=" * 50)
print("  BM25 BASELINE RESULTS")
print(f"  Queries: {len(queries)}")
print()
for k, v in metrics.items():
    print(f"  {k:<15} {v:.4f}")
print("=" * 50)
