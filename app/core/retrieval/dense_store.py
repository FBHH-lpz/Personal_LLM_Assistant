"""ChromaDB wrapper for dense vector storage and search.

Replaces Milvus Lite which had data persistence issues on Windows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks"
IMAGE_COLLECTION = "rag_images"  # Separate collection for image descriptions


class DenseStore:
    """ChromaDB vector store for dense embeddings.

    Usage::

        store = DenseStore(db_path="./data/chroma_db")
        await store.ensure_collection()
        store.insert(ids, embeddings, metadata, documents)
        results = store.search(query_embedding, top_k=10)
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None

    # ── lifecycle ──────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Get or create the collection. Idempotent."""
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
            logger.info("ChromaDB collection '%s': %d docs",
                        COLLECTION_NAME, self._collection.count())
        except Exception:
            self._collection = self._client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection '%s' created", COLLECTION_NAME)

    # ── CRUD ───────────────────────────────────────────────────

    def insert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict],
    ) -> None:
        """Insert vectors with metadata into ChromaDB."""
        if not ids or self._collection is None:
            return

        documents = [m.get("child_id", ids[i]) for i, m in enumerate(metadata_list)]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata_list,
            documents=documents,
        )
        logger.debug("Inserted %d vectors into ChromaDB", len(ids))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Search for nearest neighbors.

        Returns list of dicts with keys: id, child_id, parent_id, source, score.
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        kwargs = {}
        if filter_expr:
            kwargs["where"] = {"source": filter_expr}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            **kwargs,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        hits = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0
            # Convert cosine distance to similarity score
            score = 1.0 - distance if distance is not None else 0.0

            hits.append({
                "id": ids[i],
                "child_id": meta.get("child_id", ids[i]),
                "parent_id": meta.get("parent_id", ""),
                "source": meta.get("source", ""),
                "score": score,
            })

        return hits

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source file."""
        if self._collection is None:
            return 0
        try:
            results = self._collection.get(where={"source": source})
            ids = results["ids"]
            if ids:
                self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for source '%s'", len(ids), source)
            return len(ids)
        except Exception:
            return 0

    async def count(self) -> int:
        """Total number of vectors in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()

    # ── Image collection (multi-vector indexing) ────────────────

    async def ensure_image_collection(self) -> None:
        """Get or create the image descriptions collection."""
        try:
            self._img_col = self._client.get_collection(IMAGE_COLLECTION)
        except Exception:
            self._img_col = self._client.create_collection(
                name=IMAGE_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB image collection '%s' created", IMAGE_COLLECTION)

    def insert_images(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict],
    ) -> None:
        """Insert image description vectors."""
        if not ids or not hasattr(self, '_img_col') or self._img_col is None:
            return
        documents = [m.get("child_id", ids[i]) for i, m in enumerate(metadata_list)]
        self._img_col.add(ids=ids, embeddings=embeddings, metadatas=metadata_list, documents=documents)
        logger.debug("Inserted %d image vectors", len(ids))

    async def search_images(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Search image descriptions."""
        if not hasattr(self, '_img_col') or self._img_col is None or self._img_col.count() == 0:
            return []
        results = self._img_col.query(query_embeddings=[query_embedding], n_results=min(top_k, self._img_col.count()))
        if not results["ids"] or not results["ids"][0]:
            return []
        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0.0
            hits.append({
                "id": results["ids"][0][i],
                "child_id": meta.get("child_id", ""),
                "parent_id": meta.get("parent_id", ""),
                "source": meta.get("source", ""),
                "type": meta.get("type", "image"),
                "score": 1.0 - dist,
            })
        return hits

    async def close(self) -> None:
        """Release resources (no-op for ChromaDB)."""
        pass
