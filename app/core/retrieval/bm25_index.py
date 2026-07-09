"""BM25 sparse index using rank-bm25 with Chinese text support."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    For Chinese text, tries jieba segmentation; falls back to character bigrams.
    For mixed CJK/Latin text, handles both.
    """
    # Try jieba for proper Chinese word segmentation
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        pass

    # Fallback: character-level with bigrams for CJK, whitespace-split for Latin
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            # CJK character: make bigrams
            if i + 1 < len(text) and ('一' <= text[i + 1] <= '鿿' or '㐀' <= text[i + 1] <= '䶿'):
                tokens.append(text[i:i + 2])
            else:
                tokens.append(ch)
            i += 1
        elif ch.isalnum():
            # Latin word: collect until non-alnum
            start = i
            while i < len(text) and text[i].isalnum():
                i += 1
            tokens.append(text[start:i].lower())
        else:
            # Punctuation/whitespace: skip
            i += 1
    return tokens


class BM25Index:
    """In-memory BM25 sparse index with optional persistence.

    Usage::

        index = BM25Index()
        index.index_documents(["doc1 text", "doc2 text"], ["id1", "id2"])
        results = index.search("query text", top_k=10)
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        # Parent storage: parent_id → (content, metadata)
        self._parents: dict[str, tuple[str, dict]] = {}

    # ── indexing ───────────────────────────────────────────────

    def index_documents(self, texts: list[str], doc_ids: list[str]) -> None:
        """Add documents to the BM25 index.

        If there are already documents, the index is rebuilt incrementally.
        """
        tokenized = [_tokenize(t) for t in texts]
        self._doc_texts.extend(texts)
        self._doc_ids.extend(doc_ids)

        if self._bm25 is None:
            self._bm25 = BM25Okapi(tokenized)
        else:
            # Rebuild index with all documents
            all_tokenized = [_tokenize(t) for t in self._doc_texts]
            self._bm25 = BM25Okapi(all_tokenized)

        logger.info("BM25 index now has %d documents", len(self._doc_ids))

    def store_parent(self, parent_id: str, content: str, metadata: dict | None = None) -> None:
        """Store a parent chunk for later context retrieval."""
        self._parents[parent_id] = (content, metadata or {})

    def get_parent(self, parent_id: str) -> tuple[str, dict] | None:
        """Retrieve a parent chunk by ID."""
        return self._parents.get(parent_id)

    # ── search ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search BM25 and return (doc_id, score) pairs sorted by relevance."""
        if self._bm25 is None or not self._doc_ids:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Pair scores with doc IDs, sort descending
        ranked = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [(doc_id, float(score)) for doc_id, score in ranked]

    # ── persistence ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save BM25 state to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "doc_ids": self._doc_ids,
            "doc_texts": self._doc_texts,
            "parents": self._parents,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved BM25 index to %s (%d docs)", path, len(self._doc_ids))

    def load(self, path: str | Path) -> bool:
        """Load BM25 state from disk. Returns False if file not found."""
        path = Path(path)
        if not path.exists():
            logger.warning("BM25 index file not found: %s", path)
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._doc_ids = data["doc_ids"]
        self._doc_texts = data["doc_texts"]
        self._parents = data.get("parents", {})

        # Rebuild BM25Okapi from loaded texts
        tokenized = [_tokenize(t) for t in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)

        logger.info("Loaded BM25 index from %s (%d docs)", path, len(self._doc_ids))
        return True

    def __len__(self) -> int:
        return len(self._doc_ids)
