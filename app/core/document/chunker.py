"""Parent-Child Chunking strategy with table-aware splitting.

Produces a two-layer structure:
- **Parent chunks** (large, ~800 tokens): full context for LLM generation.
- **Child chunks** (small, ~200 tokens): fine-grained units for vector search.

Tables are detected and kept as atomic units — never split mid-table.
"""

from __future__ import annotations

import re
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


# Pattern to detect Markdown table rows: starts with |, ends with |,
# contains at least one more | (so it's a table, not a single-cell line)
_TABLE_ROW = re.compile(r'^\|.+\|.+\|.*$')
# Separator line: |---|----|
_TABLE_SEP = re.compile(r'^\|[\s\-:]+\|[\s\-:]+\|.*$')


class ParentChildChunker:
    """Split text into parent (large) and child (small) chunks.

    Table-aware: Markdown tables (from Docling) are detected and kept as
    atomic child chunks — never broken across token boundaries.

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

        self.parent_size = parent_size
        self.child_size = child_size
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

    # ── table detection & protection ───────────────────────────

    def _find_tables(self, text: str) -> list[tuple[int, int, str]]:
        """Find Markdown table blocks. Returns [(start, end, table_text), ...]."""
        lines = text.split('\n')
        tables: list[tuple[int, int, str]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if _TABLE_ROW.match(line) or _TABLE_SEP.match(line):
                start = i
                # Walk back to capture the caption/header line before the table
                if start > 0 and not lines[start - 1].strip():
                    start -= 1
                # Walk forward to capture all table rows
                while i < len(lines) and (_TABLE_ROW.match(lines[i]) or
                                          _TABLE_SEP.match(lines[i]) or
                                          not lines[i].strip()):
                    i += 1
                table_text = '\n'.join(lines[start:i])
                tables.append((start, i, table_text))
            else:
                i += 1
        return tables

    def _protect_tables(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace table blocks with placeholders. Returns (protected_text, table_map)."""
        tables = self._find_tables(text)
        if not tables:
            return text, {}

        lines = text.split('\n')
        table_map: dict[str, str] = {}
        # Process in reverse to preserve line indices
        for idx, (start, end, table_text) in enumerate(reversed(tables)):
            placeholder = f'\n[TABLE_{len(tables) - 1 - idx}]\n'
            table_map[placeholder.strip()] = table_text
            # Replace table lines with placeholder
            lines[start:end] = [placeholder]
        return '\n'.join(lines), table_map

    def _restore_tables(self, text: str, table_map: dict[str, str]) -> str:
        """Replace placeholders back with original tables."""
        result = text
        for placeholder, table_text in table_map.items():
            # Placeholder might be embedded in a chunk with surrounding text
            result = result.replace(placeholder, f'\n{table_text}\n')
        return result

    # ── main entry ─────────────────────────────────────────────

    def chunk(self, text: str, source_metadata: dict | None = None) -> list[ChunkGroup]:
        """Split text into parent-child chunk groups with table protection.

        Args:
            text: The full document text (typically Docling markdown output).
            source_metadata: Optional metadata to attach to each group.

        Returns:
            A list of ChunkGroups, each containing one parent and its children.
            Tables are kept as atomic child chunks.
        """
        meta = source_metadata or {}

        # 1. Protect tables from being split
        protected_text, table_map = self._protect_tables(text)

        # 2. Split into parents
        parents = self.parent_splitter.split_text(protected_text)
        groups: list[ChunkGroup] = []

        for i, parent_text in enumerate(parents):
            parent_id = f"p_{i}"

            # 3. Restore tables in parent
            parent_full = self._restore_tables(parent_text, table_map)

            # 4. Split parent into children (table-aware)
            children = self._split_children_table_aware(parent_full, parent_id)

            groups.append(ChunkGroup(
                parent_id=parent_id,
                parent_content=parent_full,
                children=children,
                metadata={**meta, "parent_index": i},
            ))

        return groups

    def _split_children_table_aware(
        self, parent_text: str, parent_id: str,
    ) -> list[ChildChunk]:
        """Split parent into children, keeping tables as atomic units."""
        # Extract tables again from the parent (after restoration they're back)
        tables = self._find_tables(parent_text)

        if not tables:
            # No tables — normal splitting
            child_texts = self.child_splitter.split_text(parent_text)
            return [
                ChildChunk(id=f"{parent_id}_c_{j}", content=ct, parent_id=parent_id)
                for j, ct in enumerate(child_texts)
            ]

        # Has tables — split non-table parts, then insert table chunks
        children: list[ChildChunk] = []
        child_idx = 0
        prev_end = 0
        lines = parent_text.split('\n')

        for start, end, table_text in tables:
            # Split the text segment before this table
            pre_text = '\n'.join(lines[prev_end:start]).strip()
            if pre_text:
                for seg in self.child_splitter.split_text(pre_text):
                    children.append(ChildChunk(
                        id=f"{parent_id}_c_{child_idx}", content=seg,
                        parent_id=parent_id,
                    ))
                    child_idx += 1

            # Insert table as a single atomic child chunk
            children.append(ChildChunk(
                id=f"{parent_id}_c_{child_idx}", content=table_text,
                parent_id=parent_id,
            ))
            child_idx += 1
            prev_end = end

        # Split the remaining text after the last table
        post_text = '\n'.join(lines[prev_end:]).strip()
        if post_text:
            for seg in self.child_splitter.split_text(post_text):
                children.append(ChildChunk(
                    id=f"{parent_id}_c_{child_idx}", content=seg,
                    parent_id=parent_id,
                ))
                child_idx += 1

        return children
