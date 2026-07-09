"""Retrieval evaluation metrics: Hit@k, MRR, NDCG, Recall@k."""

from __future__ import annotations

import math
from typing import Sequence


def hit_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 5) -> bool:
    """Did we get at least one relevant doc in the top-k?"""
    return bool(set(retrieved[:k]) & relevant)


def hit_rate(relevant_list: list[set[str]], retrieved_list: list[list[str]], k: int = 5) -> float:
    """Proportion of queries with at least one hit in top-k."""
    if not relevant_list:
        return 0.0
    hits = sum(1 for rel, ret in zip(relevant_list, retrieved_list) if hit_at_k(rel, ret, k))
    return hits / len(relevant_list)


def mrr(relevant_list: list[set[str]], retrieved_list: list[list[str]]) -> float:
    """Mean Reciprocal Rank: average of 1/rank_of_first_relevant."""
    if not relevant_list:
        return 0.0

    reciprocal_ranks: list[float] = []
    for rel, ret in zip(relevant_list, retrieved_list):
        for i, doc_id in enumerate(ret):
            if doc_id in rel:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 5) -> float:
    """Proportion of all relevant docs that appear in top-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def avg_recall(relevant_list: list[set[str]], retrieved_list: list[list[str]], k: int = 5) -> float:
    """Average recall@k across queries."""
    if not relevant_list:
        return 0.0
    recalls = [recall_at_k(rel, ret, k) for rel, ret in zip(relevant_list, retrieved_list)]
    return sum(recalls) / len(recalls)


def ndcg_at_k(relevant: set[str], retrieved: Sequence[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0 for rank 1

    # IDCG (ideal: all relevant docs at the top)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def avg_ndcg(relevant_list: list[set[str]], retrieved_list: list[list[str]], k: int = 5) -> float:
    """Average NDCG@k across queries."""
    if not relevant_list:
        return 0.0
    scores = [ndcg_at_k(rel, ret, k) for rel, ret in zip(relevant_list, retrieved_list)]
    return sum(scores) / len(scores)
