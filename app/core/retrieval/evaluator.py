"""Quick retrieval evaluator — lightweight version for development use.

For full evaluation with ablations, use eval/run_eval.py.
"""

from __future__ import annotations

from app.core.retrieval.hybrid_retriever import HybridRetriever
from eval.metrics import avg_ndcg, avg_recall, hit_rate, mrr


async def quick_eval(
    retriever: HybridRetriever,
    test_queries: list[dict],
    top_k: int = 5,
) -> dict:
    """Run a quick evaluation with a small set of test queries.

    Args:
        retriever: The hybrid retriever to evaluate.
        test_queries: List of dicts with 'query' and 'relevant_chunks' keys.
        top_k: Number of top results to consider.

    Returns:
        Dict with hit_at_k, mrr, recall, ndcg metrics.
    """
    retrieved_list: list[list[str]] = []
    relevant_list: list[set[str]] = []

    for item in test_queries:
        query = item["query"]
        relevant = set(item.get("relevant_chunks", []))

        docs = await retriever.search(query, top_k=top_k)
        retrieved_ids = [d.child_id for d in docs]

        retrieved_list.append(retrieved_ids)
        relevant_list.append(relevant)

    return {
        "hit_at_1": hit_rate(relevant_list, retrieved_list, k=1),
        "hit_at_5": hit_rate(relevant_list, retrieved_list, k=5),
        "hit_at_10": hit_rate(relevant_list, retrieved_list, k=10),
        "mrr": mrr(relevant_list, retrieved_list),
        "recall_at_5": avg_recall(relevant_list, retrieved_list, k=5),
        "ndcg_at_5": avg_ndcg(relevant_list, retrieved_list, k=5),
    }
