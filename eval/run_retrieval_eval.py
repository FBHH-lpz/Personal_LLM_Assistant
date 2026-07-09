#!/usr/bin/env python3
"""Retrieval eval: BM25+Dense+RRF (no CrossEncoder). Fast and reliable."""

import json, sys, asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from eval.metrics import hit_rate, mrr, avg_recall, avg_ndcg

async def main():
    # Load
    bm25 = BM25Index()
    bm25.load("data/bm25_index_v2.pkl")
    dense = DenseStore(db_path="data/milvus_eval.db")
    await dense.ensure_collection()
    embedder = get_embedding_provider()
    retriever = HybridRetriever(dense, bm25, embedder)

    with open("eval/dataset.jsonl", encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]

    print(f"BM25: {len(bm25)} docs | Dataset: {len(queries)} queries")

    # BM25 only
    bm25_ids, rel_sets = [], []
    for i, item in enumerate(queries):
        r = bm25.search(item["query"], top_k=10)
        bm25_ids.append([x[0] for x in r])
        rel_sets.append(set(item.get("relevant_chunks", [])))
        if (i+1) % 200 == 0: print(f"  BM25: {i+1}/{len(queries)}")

    # BM25+Dense+RRF
    hybrid_ids = []
    for i, item in enumerate(queries):
        docs = await retriever.search(item["query"], top_k=10)
        hybrid_ids.append([d.child_id for d in docs])
        if (i+1) % 100 == 0: print(f"  Hybrid: {i+1}/{len(queries)}")

    def calc(ids):
        return {
            "hit_at_1": hit_rate(rel_sets, ids, k=1),
            "hit_at_3": hit_rate(rel_sets, ids, k=3),
            "hit_at_5": hit_rate(rel_sets, ids, k=5),
            "hit_at_10": hit_rate(rel_sets, ids, k=10),
            "mrr": mrr(rel_sets, ids),
            "recall_at_5": avg_recall(rel_sets, ids, k=5),
            "ndcg_at_5": avg_ndcg(rel_sets, ids, k=5),
        }

    bm = calc(bm25_ids)
    hb = calc(hybrid_ids)

    print("\n" + "=" * 65)
    print("  RETRIEVAL EVALUATION RESULTS (400 queries)")
    print(f"  {'Metric':<15} {'BM25':<15} {'BM25+Dense+RRF':<18} {'Delta':<10}")
    print(f"  {'-'*15} {'-'*15} {'-'*18} {'-'*10}")
    for k in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "recall_at_5", "ndcg_at_5"]:
        d = hb[k] - bm[k]
        print(f"  {k:<15} {bm[k]:<15.4f} {hb[k]:<18.4f} {d:+.4f}")
    print("=" * 65)

    # Save
    import json as j
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "retrieval_eval.json", "w") as f:
        j.dump({"bm25": bm, "hybrid_rrf": hb}, f, indent=2)
    print("\nSaved to eval/results/retrieval_eval.json")

asyncio.run(main())
