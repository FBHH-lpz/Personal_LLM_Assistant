#!/usr/bin/env python3
"""One-shot: ingest + eval in the same process so Milvus doesn't restart."""

import asyncio, json, logging, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.document.ingestor import DocumentIngestor
from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker
from eval.metrics import hit_rate, mrr, avg_recall, avg_ndcg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    # ── 1. Ingest ─────────────────────────────────────────
    logger.info("=== Phase 1: Ingestion ===")
    dense = DenseStore(db_path="data/milvus_eval.db")
    await dense.ensure_collection()
    bm25 = BM25Index()
    embedder = get_embedding_provider()
    ingestor = DocumentIngestor(milvus_store=dense, bm25_index=bm25)

    lecture_dir = Path("lecture")
    files = sorted(lecture_dir.glob("*.pdf"))
    logger.info("Ingesting %d files...", len(files))

    for fp in files:
        try:
            await ingestor.ingest_file(fp)
        except Exception:
            logger.exception("Failed: %s", fp.name)

    logger.info("Ingested: %d BM25 docs, %d parents", len(bm25), len(bm25._parents))

    # ── 2. Load dataset ────────────────────────────────────
    logger.info("=== Phase 2: Load Eval Dataset ===")
    ds_path = Path("eval/dataset.jsonl")
    if not ds_path.exists():
        logger.error("Dataset not found. Run generate_dataset.py first.")
        return

    queries = []
    with open(ds_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    logger.info("Loaded %d queries", len(queries))

    # ── 3. Evaluate (retrieval only, fast) ─────────────────
    logger.info("=== Phase 3: Retrieval Eval (BM25+Dense+RRF) ===")
    retriever = HybridRetriever(dense, bm25, embedder)

    ret_ids: list[list[str]] = []
    rel_sets: list[set[str]] = []

    for i, item in enumerate(queries):
        query = item["query"]
        relevant = set(item.get("relevant_chunks", []))
        docs = await retriever.search(query, top_k=10)
        ret_ids.append([d.child_id for d in docs])
        rel_sets.append(relevant)
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d queries done", i + 1, len(queries))

    ret_metrics = {
        "hit_at_1": hit_rate(rel_sets, ret_ids, k=1),
        "hit_at_3": hit_rate(rel_sets, ret_ids, k=3),
        "hit_at_5": hit_rate(rel_sets, ret_ids, k=5),
        "hit_at_10": hit_rate(rel_sets, ret_ids, k=10),
        "mrr": mrr(rel_sets, ret_ids),
        "recall_at_5": avg_recall(rel_sets, ret_ids, k=5),
        "ndcg_at_5": avg_ndcg(rel_sets, ret_ids, k=5),
    }

    # ── 4. Evaluate (with CrossEncoder) ────────────────────
    logger.info("=== Phase 4: Full Eval (BM25+Dense+RRF+CrossEncoder) ===")
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        device=settings.reranker_device,
    )
    await reranker.ensure_loaded()

    rerank_ids: list[list[str]] = []

    for i, item in enumerate(queries):
        query = item["query"]
        relevant = set(item.get("relevant_chunks", []))
        docs = await retriever.search(query, top_k=20)
        if docs:
            docs = await reranker.rerank_retrieved(query, docs, top_k=5)
        rerank_ids.append([d.child_id for d in docs])
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d queries done", i + 1, len(queries))

    ce_metrics = {
        "hit_at_1": hit_rate(rel_sets, rerank_ids, k=1),
        "hit_at_3": hit_rate(rel_sets, rerank_ids, k=3),
        "hit_at_5": hit_rate(rel_sets, rerank_ids, k=5),
        "hit_at_10": hit_rate(rel_sets, rerank_ids, k=10),
        "mrr": mrr(rel_sets, rerank_ids),
        "recall_at_5": avg_recall(rel_sets, rerank_ids, k=5),
        "ndcg_at_5": avg_ndcg(rel_sets, rerank_ids, k=5),
    }

    # ── 5. Print Results ───────────────────────────────────
    print()
    print("=" * 60)
    print("  RAG RETRIEVAL EVALUATION RESULTS")
    print(f"  Dataset: {len(queries)} queries, {sum(len(r) for r in rel_sets)} relevant chunks")
    print()
    print(f"  {'Metric':<15} {'Retrieval(RRF)':<18} {'+CrossEncoder':<18}")
    print(f"  {'-'*15} {'-'*18} {'-'*18}")
    for key in ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr", "recall_at_5", "ndcg_at_5"]:
        print(f"  {key:<15} {ret_metrics[key]:<18.4f} {ce_metrics[key]:<18.4f}")
    print("=" * 60)

    # Save
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": ts,
        "num_queries": len(queries),
        "retrieval_only": ret_metrics,
        "with_cross_encoder": ce_metrics,
    }
    out = results_dir / f"full_eval_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    asyncio.run(main())
