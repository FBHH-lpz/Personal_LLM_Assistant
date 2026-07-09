"""Admin routes — health check and system info."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import _get_bm25_index, _get_dense_store

router = APIRouter(tags=["admin"])


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "ok"}


@router.get("/stats")
async def system_stats():
    """System statistics."""
    bm25 = _get_bm25_index()
    dense = _get_dense_store()

    bm25_count = len(bm25)
    dense_count = await dense.count()

    # Count images
    img_count = 0
    if hasattr(dense, '_img_col') and dense._img_col is not None:
        img_count = dense._img_col.count()

    return {
        "bm25_documents": bm25_count,
        "dense_vectors": dense_count,
        "image_vectors": img_count,
        "parents_stored": len(bm25._parents) if hasattr(bm25, "_parents") else 0,
    }
