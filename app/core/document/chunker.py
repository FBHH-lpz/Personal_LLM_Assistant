"""Parent-Child Chunking strategy.

Produces a two-layer structure:
- **Parent chunks** (large, ~800 tokens): full context for LLM generation.
- **Child chunks** (small, ~200 tokens): fine-grained units for vector search.

Search is done against child embeddings; retrieval returns the parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ChildChunk:
    """A small searchable chunk."""
    id: str
    content: str
    parent_id: str = ""


@dataclass
class ChunkGroup:
    """A parent with its children."""
    parent_id: str
    parent_content: str
    children: list[ChildChunk]
    metadata: dict = field(default_factory=dict)


class ParentChildChunker:
    """Split text into parent (large) and child (small) chunks.

    Uses RecursiveCharacterTextSplitter from langchain for both layers.

    Usage::

        chunker = ParentChildChunker(parent_size=800, child_size=200)
        groups = chunker.chunk("long document text...")
        for g in groups:
            print(g.parent_content)          # large chunk for LLM
            for c in g.children:
                print(c.id, c.content)       # small chunks for search
    """

    def __init__(
        self,
        parent_size: int = 800,
        child_size: int = 200,
        parent_overlap: int = 25,
        child_overlap: int = 50,
    ):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def chunk(self, text: str, source_metadata: dict | None = None) -> list[ChunkGroup]:
        """Split text into parent-child chunk groups.

        Args:
            text: The full document text.
            source_metadata: Optional metadata to attach to each group.

        Returns:
            A list of ChunkGroups, each containing one parent and its children.
        """
        meta = source_metadata or {}
        parents = self.parent_splitter.split_text(text)
        groups: list[ChunkGroup] = []

        for i, parent_text in enumerate(parents):
            parent_id = f"p_{i}"
            children_texts = self.child_splitter.split_text(parent_text)
            children = [
                ChildChunk(
                    id=f"{parent_id}_c_{j}",
                    content=ct,
                    parent_id=parent_id,
                )
                for j, ct in enumerate(children_texts)
            ]
            groups.append(ChunkGroup(
                parent_id=parent_id,
                parent_content=parent_text,
                children=children,
                metadata={**meta, "parent_index": i},
            ))

        return groups
