"""Assembler chunker - splits by labels."""
from __future__ import annotations

import logging
import re
from typing import List

from src.model import Chunk, Document

logger = logging.getLogger(__name__)

# A label starts at column 0, consists of [A-Za-z_.][A-Za-z0-9_.]* and ends with `:`.
LABEL_RE = re.compile(r"^([A-Za-z_.][\w.]*):\s*(?:;.*)?$", re.MULTILINE)


class AsmChunker:
    """Chunks assembler source by labels.

    Each chunk contains a label and all instructions/directives up to the
    next label.
    """

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        matches = list(LABEL_RE.finditer(text))

        if not matches:
            return self._single(document)

        chunks: List[Chunk] = []

        # Optional prefix (directives/comments before first label)
        if matches[0].start() > 0:
            preface = text[: matches[0].start()].strip()
            if preface:
                chunks.append(
                    Chunk(
                        text=preface,
                        source_file=str(document.file_path),
                        category=document.category,
                        metadata={
                            "label": "(file_header)",
                            "source_file": document.file_name,
                        },
                    )
                )

        for i, match in enumerate(matches):
            label = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if not block:
                continue
            chunks.append(
                Chunk(
                    text=block,
                    source_file=str(document.file_path),
                    category=document.category,
                    metadata={
                        "label": label,
                        "source_file": document.file_name,
                    },
                )
            )

        logger.debug("ASM chunker produced %d chunks for %s", len(chunks), document.file_name)
        return chunks

    @staticmethod
    def _single(document: Document) -> List[Chunk]:
        if not document.content.strip():
            return []
        return [
            Chunk(
                text=document.content.strip(),
                source_file=str(document.file_path),
                category=document.category,
                metadata={
                    "label": "(no_labels)",
                    "source_file": document.file_name,
                },
            )
        ]
