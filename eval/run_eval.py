#!/usr/bin/env python3
"""Retrieval evaluation runner.

Usage:
    python eval/run_eval.py --config hybrid_rrf_ce
    python eval/run_eval.py --compare eval/results/b3.json eval/results/b4.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.llm.registry import get_embedding_provider
from app.core.retrieval.bm25_index import BM25Index
from app.core.retrieval.dense_store import DenseStore
from app.core.retrieval.hybrid_retriever import HybridRetriever
from app.core.retrieval.reranker import CrossEncoderReranker
from eval.metrics import avg_ndcg, avg_recall, hit_rate, mrr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EvalRunner:
    """Runs retrieval evaluation on a labeled dataset."""

    def __init__(self, dataset_path: Path):
        self.dataset = self._load_dataset(dataset_path)
        logger.info("Loaded %d eval queries from %s", len(self.dataset), dataset_path)

    def _load_dataset(self, path: Path) -> list[dict]:
        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(json.loads(line))
        return queries

    async def evaluate_retriever(
        self,
        retriever: HybridRetriever,
        top_k: int = 5,
    ) -> dict:
        """Evaluate a retriever on the dataset."""
        retrieved_list: list[list[str]] = []
        relevant_list: list[set[str]] = []

        for item in self.dataset:
            query = item.get("rewritten_query") or item["query"]
            relevant = set(item.get("relevant_chunks", []))

            docs = await retriever.search(query, top_k=top_k)
            retrieved_ids = [d.child_id for d in docs]

            retrieved_list.append(retrieved_ids)
            relevant_list.append(relevant)

        return {
            "hit_at_1": hit_rate(relevant_list, retrieved_list, k=1),
            "hit_at_3": hit_rate(relevant_list, retrieved_list, k=3),
            "hit_at_5": hit_rate(relevant_list, retrieved_list, k=5),
            "hit_at_10": hit_rate(relevant_list, retrieved_list, k=10),
            "mrr": mrr(relevant_list, retrieved_list),
            "recall_at_5": avg_recall(relevant_list, retrieved_list, k=5),
            "ndcg_at_5": avg_ndcg(relevant_list, retrieved_list, k=5),
            "total_queries": len(self.dataset),
            "total_relevant": sum(len(r) for r in relevant_list),
        }

    async def evaluate_with_reranker(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> dict:
        """Evaluate full pipeline: retrieval + reranking."""
        retrieved_list: list[list[str]] = []
        relevant_list: list[set[str]] = []

        for item in self.dataset:
            query = item.get("rewritten_query") or item["query"]
            relevant = set(item.get("relevant_chunks", []))

            # Retrieve
            docs = await retriever.search(query, top_k=retrieval_top_k)

            # Rerank
            if docs:
                contents = [d.content for d in docs]
                scored = await reranker.rerank(query, contents, top_k=rerank_top_k)
                content_to_score = {text: score for text, score in scored}
                docs.sort(key=lambda d: content_to_score.get(d.content, 0.0), reverse=True)
                docs = docs[:rerank_top_k]

            retrieved_ids = [d.child_id for d in docs]
            retrieved_list.append(retrieved_ids)
            relevant_list.append(relevant)

        return {
            "hit_at_1": hit_rate(relevant_list, retrieved_list, k=1),
            "hit_at_3": hit_rate(relevant_list, retrieved_list, k=3),
            "hit_at_5": hit_rate(relevant_list, retrieved_list, k=5),
            "hit_at_10": hit_rate(relevant_list, retrieved_list, k=10),
            "mrr": mrr(relevant_list, retrieved_list),
            "recall_at_5": avg_recall(relevant_list, retrieved_list, k=5),
            "ndcg_at_5": avg_ndcg(relevant_list, retrieved_list, k=5),
            "total_queries": len(self.dataset),
            "total_relevant": sum(len(r) for r in relevant_list),
        }


def print_results(metrics: dict, config_name: str = ""):
    """Pretty-print evaluation metrics."""
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Queries: {metrics['total_queries']}, Total relevant: {metrics['total_relevant']}")
    print()
    print(f"  Hit@1:   {metrics['hit_at_1']:.4f}")
    print(f"  Hit@3:   {metrics['hit_at_3']:.4f}")
    print(f"  Hit@5:   {metrics['hit_at_5']:.4f}  ← 主要指标")
    print(f"  Hit@10:  {metrics['hit_at_10']:.4f}")
    print(f"  MRR:     {metrics['mrr']:.4f}")
    print(f"  Recall@5:{metrics['recall_at_5']:.4f}")
    print(f"  NDCG@5:  {metrics['ndcg_at_5']:.4f}")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="RAG Retrieval Evaluation")
    parser.add_argument("--dataset", default="eval/dataset.jsonl", help="Path to eval dataset")
    parser.add_argument("--config", default="hybrid_rrf_ce", help="Config name for output")
    parser.add_argument("--compare", nargs="*", help="Compare two result JSON files")
    parser.add_argument("--mode", default="full", choices=["retrieval", "full"],
                        help="Evaluate retrieval only or full pipeline with reranker")
    args = parser.parse_args()

    # Compare mode
    if args.compare and len(args.compare) == 2:
        with open(args.compare[0]) as f:
            r1 = json.load(f)
        with open(args.compare[1]) as f:
            r2 = json.load(f)
        print_results(r1, args.compare[0])
        print_results(r2, args.compare[1])
        print(f"\nHit@5 delta: {r2['hit_at_5'] - r1['hit_at_5']:+.4f}")
        print(f"MRR delta:   {r2['mrr'] - r1['mrr']:+.4f}")
        return

    # Evaluation mode
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    # Initialize retriever
    embedder = get_embedding_provider()
    dense = DenseStore(db_path=settings.milvus_db_path)
    bm25 = BM25Index()
    bm25.load(settings.bm25_index_path)
    retriever = HybridRetriever(dense, bm25, embedder)

    runner = EvalRunner(dataset_path)

    if args.mode == "retrieval":
        metrics = await runner.evaluate_retriever(retriever, top_k=5)
    else:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
        )
        metrics = await runner.evaluate_with_reranker(retriever, reranker)

    print_results(metrics, args.config)

    # Save results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    asyncio.run(main())
