"""Markdown chunker - splits documents by headings."""
from __future__ import annotations

import logging
import re
from typing import List

from src.model import Chunk, Document

logger = logging.getLogger(__name__)

# Matches ATX-style headings (# ... ######) at the start of a line.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class MarkdownChunker:
    """Chunks Markdown documents along headings.

    Each chunk corresponds to one heading + its following body until the
    next heading at the same or higher level.
    """

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        matches = list(HEADING_RE.finditer(text))

        if not matches:
            # No headings -> single chunk
            return self._single_chunk(document)

        chunks: List[Chunk] = []
        # Include any prefix before the first heading
        if matches[0].start() > 0:
            preface = text[: matches[0].start()].strip()
            if preface:
                chunks.append(
                    Chunk(
                        text=preface,
                        source_file=str(document.file_path),
                        category=document.category,
                        metadata={
                            "heading": "(preface)",
                            "level": 0,
                            "source_file": document.file_name,
                        },
                    )
                )

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_file=str(document.file_path),
                    category=document.category,
                    metadata={
                        "heading": heading,
                        "level": level,
                        "source_file": document.file_name,
                    },
                )
            )

        logger.debug("Markdown chunker produced %d chunks for %s", len(chunks), document.file_name)
        return chunks

    @staticmethod
    def _single_chunk(document: Document) -> List[Chunk]:
        if not document.content.strip():
            return []
        return [
            Chunk(
                text=document.content.strip(),
                source_file=str(document.file_path),
                category=document.category,
                metadata={
                    "heading": "(full document)",
                    "level": 0,
                    "source_file": document.file_name,
                },
            )
        ]
