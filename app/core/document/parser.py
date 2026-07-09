"""Document parsers: PDF (PyMuPDF), Word (python-docx), TXT, Markdown."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Parsed document with metadata."""
    text: str
    filename: str
    page_count: int = 1
    metadata: dict | None = None
    # Image-rich pages: (page_number, image_path, page_text)
    image_pages: list[dict] = field(default_factory=list)


def parse_pdf(filepath: Path, extract_images: bool = True) -> ParsedDocument:
    """Extract text from a PDF using PyMuPDF (fitz).

    When extract_images=True, pages with embedded images are rendered
    as PNG files for later VLM analysis.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(filepath))
    num_pages = doc.page_count
    pages: list[str] = []
    image_pages: list[dict] = []

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append(text)

        # Detect embedded images on this page
        if extract_images:
            image_list = page.get_images(full=True)
            if image_list:
                # Render page as high-res image for VLM
                mat = fitz.Matrix(2.0, 2.0)  # 2x resolution
                pix = page.get_pixmap(matrix=mat)
                img_path = filepath.parent / f"_img_{filepath.stem}_p{page_num + 1}.png"
                pix.save(str(img_path))

                image_pages.append({
                    "page_number": page_num + 1,
                    "image_path": str(img_path),
                    "image_count": len(image_list),
                    "page_text": text.strip()[:500],  # context for description
                })

    doc.close()
    full_text = "\n\n".join(pages)
    logger.info(
        "Parsed PDF '%s': %d pages, %d chars, %d image pages",
        filepath.name, num_pages, len(full_text), len(image_pages),
    )
    return ParsedDocument(
        text=full_text,
        filename=filepath.name,
        page_count=num_pages,
        metadata={"source": str(filepath), "type": "pdf"},
        image_pages=image_pages,
    )


def parse_docx(filepath: Path) -> ParsedDocument:
    """Extract text from a Word document."""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(filepath))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)
    logger.info("Parsed DOCX '%s': %d paragraphs, %d chars", filepath.name, len(paragraphs), len(full_text))
    return ParsedDocument(
        text=full_text,
        filename=filepath.name,
        metadata={"source": str(filepath), "type": "docx"},
    )


def parse_txt(filepath: Path) -> ParsedDocument:
    """Read a plain text file."""
    text = filepath.read_text(encoding="utf-8")
    logger.info("Read TXT '%s': %d chars", filepath.name, len(text))
    return ParsedDocument(
        text=text,
        filename=filepath.name,
        metadata={"source": str(filepath), "type": "txt"},
    )


def parse_markdown(filepath: Path) -> ParsedDocument:
    """Read a markdown file."""
    text = filepath.read_text(encoding="utf-8")
    logger.info("Read MD '%s': %d chars", filepath.name, len(text))
    return ParsedDocument(
        text=text,
        filename=filepath.name,
        metadata={"source": str(filepath), "type": "markdown"},
    )


# ── dispatcher ────────────────────────────────────────────────

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".doc": parse_docx,
    ".txt": parse_txt,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
}


def parse_document(filepath: Path, extract_images: bool = True) -> ParsedDocument:
    """Parse any supported document type."""
    ext = filepath.suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(PARSERS.keys())}")
    if ext == ".pdf":
        return parser(filepath, extract_images=extract_images)
    return parser(filepath)
