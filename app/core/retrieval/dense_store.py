"""Milvus Lite wrapper for dense vector storage and search."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
)
from pymilvus.milvus_client.index import IndexParams

logger = logging.getLogger(__name__)

# Default embedding dimension for Tongyi text-embedding-v3
DEFAULT_DIM = 1024
COLLECTION_NAME = "rag_chunks"


class DenseStore:
    """Milvus Lite vector store for dense embeddings.

    Uses MilvusClient (simpler API, no need for connect/load/query dance).

    Usage::

        store = DenseStore(db_path="./data/milvus.db", dim=1024)
        await store.ensure_collection()
        await store.insert(ids, embeddings, metadata)
        results = await store.search(query_vector, top_k=10)
    """

    def __init__(self, db_path: str, dim: int = DEFAULT_DIM):
        self.db_path = db_path
        self.dim = dim
        self._client: Optional[MilvusClient] = None
        self._ready = False

    # ── lifecycle ──────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        client = self._get_client()

        if client.has_collection(COLLECTION_NAME):
            logger.info("Milvus collection '%s' already exists", COLLECTION_NAME)
            client.load_collection(COLLECTION_NAME)
            self._ready = True
            return

        # Define schema
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="child_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            ],
            description="RAG document chunks with embeddings",
        )

        # Create collection with index
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
        )

        # Create IVF_FLAT index for vector search
        index_params = IndexParams()
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_idx",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=index_params,
        )

        logger.info("Created Milvus collection '%s' (dim=%d)", COLLECTION_NAME, self.dim)
        self._ready = True

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._client = MilvusClient(uri=self.db_path)
        return self._client

    # ── CRUD ───────────────────────────────────────────────────

    async def insert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict],
    ) -> None:
        """Insert vectors with metadata into Milvus."""
        if not ids:
            return

        client = self._get_client()
        data = [
            {
                "id": ids[i],
                "embedding": embeddings[i],
                "child_id": metadata_list[i].get("child_id", ids[i]),
                "parent_id": metadata_list[i].get("parent_id", ""),
                "source": metadata_list[i].get("source", ""),
            }
            for i in range(len(ids))
        ]

        client.insert(collection_name=COLLECTION_NAME, data=data)
        logger.debug("Inserted %d vectors into Milvus", len(ids))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Search for nearest neighbors.

        Returns list of dicts with keys: id, child_id, parent_id, source, distance, embedding.
        """
        client = self._get_client()

        # Ensure collection is loaded (it may be released between sessions)
        try:
            client.load_collection(COLLECTION_NAME)
        except Exception:
            pass  # Already loaded or doesn't exist yet

        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            output_fields=["child_id", "parent_id", "source"],
        )

        if not results or not results[0]:
            return []

        # results[0] is the list of hits for the first (only) query vector
        hits = []
        for hit in results[0]:
            entity = hit.get("entity", hit)
            hits.append({
                "id": entity.get("id", ""),
                "child_id": entity.get("child_id", ""),
                "parent_id": entity.get("parent_id", ""),
                "source": entity.get("source", ""),
                "score": hit.get("distance", 0.0),
            })

        return hits

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source file. Returns count of deleted entities."""
        client = self._get_client()
        result = client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'source == "{source}"',
        )
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.info("Deleted %d chunks for source '%s'", count, source)
        return count

    async def count(self) -> int:
        """Total number of vectors in the collection."""
        client = self._get_client()
        stats = client.get_collection_stats(COLLECTION_NAME)
        return stats.get("row_count", 0)

    async def close(self) -> None:
        """Release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
