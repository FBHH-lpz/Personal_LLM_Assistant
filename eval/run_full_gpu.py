#!/usr/bin/env python3
"""Full eval: BM25 → Hybrid(RRF) → CrossEncoder — all in one process with GPU."""

import json, sys, asyncio, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

from app.core.document.ingestor import DocumentIngestor
from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker
from eval.metrics import hit_rate, mrr, avg_recall, avg_ndcg

async def main():
    # ── 1. Ingestion ───────────────────────────────────────
    print("=== Phase 1: Ingestion ===")
    dense = DenseStore(db_path="data/milvus_full.db")
    await dense.ensure_collection()
    bm25 = BM25Index()
    embedder = get_embedding_provider()
    ingestor = DocumentIngestor(milvus_store=dense, bm25_index=bm25)

    files = sorted(Path("lecture").glob("*.pdf"))
    for i, fp in enumerate(files):
        try:
            await ingestor.ingest_file(fp)
        except Exception as e:
            print(f"  FAIL: {fp.name}: {e}")
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] {fp.name}")
    print(f"Ingestion: {len(bm25)} chunks, {len(bm25._parents)} parents")

    # ── 2. Load Dataset ────────────────────────────────────
    print("\n=== Phase 2: Load Dataset ===")
    with open("eval/dataset.jsonl", encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(queries)} queries")
    rel_sets = [set(q.get("relevant_chunks", [])) for q in queries]
    query_texts = [q["query"] for q in queries]

    # ── 3. BM25 ────────────────────────────────────────────
    print("\n=== Phase 3: BM25 ===")
    bm25_ids = []
    for i, q in enumerate(query_texts):
        r = bm25.search(q, top_k=10)
        bm25_ids.append([x[0] for x in r])

    # ── 4. Hybrid (BM25+Dense+RRF) ─────────────────────────
    print("\n=== Phase 4: Hybrid (BM25+Dense+RRF) ===")
    retriever = HybridRetriever(dense, bm25, embedder)
    hybrid_ids = []
    for i, q in enumerate(query_texts):
        docs = await retriever.search(q, top_k=20)
        hybrid_ids.append([d.child_id for d in docs])
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(queries)}")

    # ── 5. CrossEncoder (GPU!) ─────────────────────────────
    print("\n=== Phase 5: CrossEncoder Rerank ===")
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cuda",
    )
    await reranker.ensure_loaded()
    print("CrossEncoder loaded on GPU")

    ce_ids = []
    for i, q in enumerate(query_texts):
        docs = await retriever.search(q, top_k=20)
        if docs:
            docs = await reranker.rerank_retrieved(q, docs, top_k=5)
        ce_ids.append([d.child_id for d in docs])
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(queries)}")

    # ── 6. Compute Metrics ─────────────────────────────────
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
    hy = calc(hybrid_ids)
    ce = calc(ce_ids)

    # ── 7. Print ───────────────────────────────────────────
    print()
    print("=" * 78)
    print("  FULL RETRIEVAL EVALUATION (GPU)")
    print(f"  {len(queries)} queries | {len(bm25)} chunks | 32 lecture PDFs")
    print()
    print(f"  {'Metric':<15} {'BM25':<10} {'Hybrid(RRF)':<14} {'+CrossEnc':<12} {'CE vs BM25':<11} {'CE vs Hyb':<10}")
    print(f"  {'-'*15} {'-'*10} {'-'*14} {'-'*12} {'-'*11} {'-'*10}")
    for k in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "recall_at_5", "ndcg_at_5"]:
        d1 = ce[k] - bm[k]
        d2 = ce[k] - hy[k]
        print(f"  {k:<15} {bm[k]:.4f}     {hy[k]:.4f}       {ce[k]:.4f}      {d1:+.4f}     {d2:+.4f}")
    print("=" * 78)

    # Save
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "full_gpu_eval.json", "w") as f:
        json.dump({"bm25": bm, "hybrid_rrf": hy, "cross_encoder": ce}, f, indent=2)
    print("\nSaved: eval/results/full_gpu_eval.json")

asyncio.run(main())
