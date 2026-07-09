"""Document upload and ingestion routes."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_embedder, get_retriever
from app.config import settings
from app.core.document.ingestor import DocumentIngestor
from app.core.document.parser import PARSERS
from app.db.models import Document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunk_count: int
    status: str
    error_message: str | None
    uploaded_at: str


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """Upload a document for ingestion into the RAG system.

    Supports: PDF, DOCX, DOC, TXT, MD, Markdown.
    """
    # Validate file type
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in PARSERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {list(PARSERS.keys())}",
        )

    # Save uploaded file to temp location
    suffix = ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    # Create DB record
    doc = Document(
        user_id=user_id,
        filename=file.filename or "unknown",
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        # Ingest
        from app.api.dependencies import _get_bm25_index, _get_dense_store

        ingestor = DocumentIngestor(
            milvus_store=_get_dense_store(),
            bm25_index=_get_bm25_index(),
        )
        groups = await ingestor.ingest_file(tmp_path)

        # Update DB
        total_children = sum(len(g.children) for g in groups)
        doc.chunk_count = total_children
        doc.status = "ready"
        await db.commit()
        await db.refresh(doc)

        logger.info("Document '%s' ingested: %d chunks", file.filename, total_children)

    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        await db.commit()
        logger.exception("Document ingestion failed: %s", file.filename)

    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except OSError:
            pass

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        chunk_count=doc.chunk_count,
        status=doc.status,
        error_message=doc.error_message,
        uploaded_at=doc.uploaded_at.isoformat() if doc.uploaded_at else "",
    )


@router.get("")
async def list_documents(
    user_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded documents."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.uploaded_at.desc())
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            chunk_count=d.chunk_count,
            status=d.status,
            error_message=d.error_message,
            uploaded_at=d.uploaded_at.isoformat() if d.uploaded_at else "",
        )
        for d in docs
    ]
