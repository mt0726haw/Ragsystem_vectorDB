"""Assembler source parser."""
from __future__ import annotations

import logging
from pathlib import Path

from src.model import Document

logger = logging.getLogger(__name__)


class AsmParser:
    """Parser for assembler source files (.asm, .s, .S)."""

    extensions = (".asm", ".s", ".S")

    def parse(self, file_path: Path, source_name: str, category: str) -> Document:
        logger.debug("Parsing asm file: %s", file_path)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return Document(
            source_name=source_name,
            category=category,
            file_path=file_path,
            file_type=file_path.suffix.lstrip(".").lower(),
            content=content,
            metadata={"size_bytes": file_path.stat().st_size},
        )
