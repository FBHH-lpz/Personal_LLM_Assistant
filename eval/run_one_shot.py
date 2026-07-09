#!/usr/bin/env python3
"""One-shot: ingest + retrieval eval in single process. No CrossEncoder."""

import json, sys, asyncio, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING)  # quiet

from app.core.document.ingestor import DocumentIngestor
from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from eval.metrics import hit_rate, mrr, avg_recall, avg_ndcg

async def main():
    print("=== Phase 1: Ingestion ===")
    dense = DenseStore(db_path="data/milvus_one_shot.db")
    await dense.ensure_collection()
    bm25 = BM25Index()
    embedder = get_embedding_provider()
    ingestor = DocumentIngestor(milvus_store=dense, bm25_index=bm25)

    files = sorted(Path("lecture").glob("*.pdf"))
    for i, fp in enumerate(files):
        try:
            await ingestor.ingest_file(fp)
        except Exception as e:
            print(f"  FAILED: {fp.name}: {e}")
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] {fp.name}")
    print(f"Ingestion done: {len(bm25)} chunks, {len(bm25._parents)} parents")

    print("\n=== Phase 2: Load Dataset ===")
    with open("eval/dataset.jsonl", encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(queries)} queries")

    print("\n=== Phase 3: BM25 Baseline ===")
    bm25_ids, rel_sets = [], []
    for i, item in enumerate(queries):
        r = bm25.search(item["query"], top_k=10)
        bm25_ids.append([x[0] for x in r])
        rel_sets.append(set(item.get("relevant_chunks", [])))

    print("\n=== Phase 4: Hybrid (BM25+Dense+RRF) ===")
    retriever = HybridRetriever(dense, bm25, embedder)
    hybrid_ids = []
    for i, item in enumerate(queries):
        docs = await retriever.search(item["query"], top_k=10)
        hybrid_ids.append([d.child_id for d in docs])
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(queries)}")

    def calc(ids):
        return {k: f(v) for k, v, f in [
            ("hit_at_1", hit_rate(rel_sets, ids, k=1), float),
            ("hit_at_3", hit_rate(rel_sets, ids, k=3), float),
            ("hit_at_5", hit_rate(rel_sets, ids, k=5), float),
            ("hit_at_10", hit_rate(rel_sets, ids, k=10), float),
            ("mrr", mrr(rel_sets, ids), float),
            ("recall_at_5", avg_recall(rel_sets, ids, k=5), float),
            ("ndcg_at_5", avg_ndcg(rel_sets, ids, k=5), float),
        ]}

    # Fix: properly compute metrics
    bm_calc = {
        "hit_at_1": hit_rate(rel_sets, bm25_ids, k=1),
        "hit_at_3": hit_rate(rel_sets, bm25_ids, k=3),
        "hit_at_5": hit_rate(rel_sets, bm25_ids, k=5),
        "hit_at_10": hit_rate(rel_sets, bm25_ids, k=10),
        "mrr": mrr(rel_sets, bm25_ids),
        "recall_at_5": avg_recall(rel_sets, bm25_ids, k=5),
        "ndcg_at_5": avg_ndcg(rel_sets, bm25_ids, k=5),
    }
    hy_calc = {
        "hit_at_1": hit_rate(rel_sets, hybrid_ids, k=1),
        "hit_at_3": hit_rate(rel_sets, hybrid_ids, k=3),
        "hit_at_5": hit_rate(rel_sets, hybrid_ids, k=5),
        "hit_at_10": hit_rate(rel_sets, hybrid_ids, k=10),
        "mrr": mrr(rel_sets, hybrid_ids),
        "recall_at_5": avg_recall(rel_sets, hybrid_ids, k=5),
        "ndcg_at_5": avg_ndcg(rel_sets, hybrid_ids, k=5),
    }

    print()
    print("=" * 65)
    print("  RETRIEVAL EVALUATION")
    print(f"  {len(queries)} queries | {len(bm25)} chunks | 32 lecture PDFs")
    print()
    print(f"  {'Metric':<15} {'BM25':<15} {'Hybrid(RRF)':<15} {'Improvement':<12}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*12}")
    for k in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "recall_at_5", "ndcg_at_5"]:
        d = hy_calc[k] - bm_calc[k]
        print(f"  {k:<15} {bm_calc[k]:.4f}         {hy_calc[k]:.4f}         {d:+.4f}")
    print("=" * 65)

    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "one_shot_eval.json", "w") as f:
        json.dump({"bm25": bm_calc, "hybrid_rrf": hy_calc}, f, indent=2)
    print("\nSaved: eval/results/one_shot_eval.json")

asyncio.run(main())
