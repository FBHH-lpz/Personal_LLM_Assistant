"""Tests for retrieval components: BM25, RRF fusion, evaluator."""

from __future__ import annotations

import pytest


class TestBM25Index:
    """Tests for BM25Index."""

    def test_index_and_search(self):
        from app.core.retrieval.bm25_index import BM25Index

        idx = BM25Index()
        texts = [
            "数据挖掘是从大量数据中提取模式的过程",
            "机器学习是人工智能的一个分支",
            "深度学习使用多层神经网络进行特征学习",
        ]
        ids = ["doc_1", "doc_2", "doc_3"]
        idx.index_documents(texts, ids)

        # Search
        results = idx.search("数据挖掘模式提取", top_k=2)
        assert len(results) > 0
        assert results[0][0] == "doc_1"  # most relevant

    def test_parent_store(self):
        from app.core.retrieval.bm25_index import BM25Index

        idx = BM25Index()
        idx.store_parent("p_1", "parent content", {"source": "test.pdf"})

        result = idx.get_parent("p_1")
        assert result is not None
        assert result[0] == "parent content"
        assert result[1]["source"] == "test.pdf"

    def test_empty_search(self):
        from app.core.retrieval.bm25_index import BM25Index

        idx = BM25Index()
        results = idx.search("anything")
        assert results == []

    def test_persistence(self, tmp_path):
        from app.core.retrieval.bm25_index import BM25Index

        idx = BM25Index()
        idx.index_documents(["test content here"], ["d1"])
        idx.store_parent("p_1", "parent text")

        save_path = tmp_path / "bm25_test.pkl"
        idx.save(str(save_path))

        # Reload
        idx2 = BM25Index()
        assert idx2.load(str(save_path))
        assert len(idx2) == 1
        results = idx2.search("test content", top_k=5)
        assert len(results) == 1
        assert results[0][0] == "d1"

        parent = idx2.get_parent("p_1")
        assert parent is not None


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_basic(self):
        from app.core.retrieval.hybrid_retriever import HybridRetriever

        # We just test the fusion function in isolation
        retriever = object.__new__(HybridRetriever)

        bm25 = [
            ("c_1", 5.0),
            ("c_2", 3.0),
            ("c_3", 1.0),
        ]
        dense = [
            {"child_id": "c_2", "score": 0.9},
            {"child_id": "c_4", "score": 0.8},
            {"child_id": "c_1", "score": 0.5},
        ]

        fused = retriever._rrf_fusion(bm25, dense, [], k=60, top_k=10)
        ids = [f[0] for f in fused]

        # c_2 appears in both lists → should rank high
        assert "c_2" in ids
        # c_1 also appears in both
        assert "c_1" in ids
        # c_4 appears only in dense, c_3 only in bm25
        assert "c_4" in ids
        assert "c_3" in ids


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_hit_at_k(self):
        from eval.metrics import hit_at_k, hit_rate

        assert hit_at_k({"a", "b", "c"}, ["a", "x", "y"], k=3) is True
        assert hit_at_k({"a", "b"}, ["x", "y", "z"], k=3) is False
        # hit_rate across multiple queries
        rate = hit_rate(
            [{"a", "b"}, {"x", "y"}],
            [["a", "z"], ["w", "v"]],
            k=3,
        )
        assert rate == 0.5

    def test_mrr(self):
        from eval.metrics import mrr

        # First query: relevant at rank 2 → 1/2 = 0.5
        # Second query: relevant at rank 1 → 1/1 = 1.0
        # Third query: no relevant → 0
        score = mrr(
            [{"a"}, {"b"}, {"x"}],
            [["z", "a", "c"], ["b"], ["y", "z"]],
        )
        assert score == pytest.approx((0.5 + 1.0 + 0.0) / 3)

    def test_recall_at_k(self):
        from eval.metrics import avg_recall, recall_at_k

        assert recall_at_k({"a", "b", "c"}, ["a", "x", "y"], k=3) == pytest.approx(1 / 3)
        recall = avg_recall(
            [{"a", "b"}, {"c", "d"}],
            [["a", "z"], ["c"]],
            k=5,
        )
        assert recall == pytest.approx(0.5)  # (0.5 + 0.5) / 2

    def test_ndcg_at_k(self):
        from eval.metrics import avg_ndcg, ndcg_at_k

        # Perfect: relevant docs at ranks 1,2
        perfect = ndcg_at_k({"a", "b"}, ["a", "b", "c"], k=3)
        assert perfect == pytest.approx(1.0)

        # Imperfect: relevant at rank 2,3
        imperfect = ndcg_at_k({"a", "b"}, ["x", "a", "b"], k=3)
        assert imperfect < 1.0
