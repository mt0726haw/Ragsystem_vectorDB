"""Chunker registry: maps content categories to chunker instances."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

from src.chunkers import AsmChunker, MarkdownChunker, PythonAstChunker, YamlChunker
from src.model import Chunk, Document

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class ChunkerRegistry:
    """Selects the right chunker based on the document category."""

    def __init__(self) -> None:
        self._chunkers: Dict[str, Any] = {
            "documentation": MarkdownChunker(),
            "python_code": PythonAstChunker(),
            "yaml_config": YamlChunker(),
            "assembler": AsmChunker(),
        }

    def register(self, category: str, chunker: Any) -> None:
        self._chunkers[category] = chunker
        logger.debug("Registered chunker for category %r -> %s", category, type(chunker).__name__)

    def get(self, category: str) -> Optional[Any]:
        return self._chunkers.get(category)

    def supported_categories(self) -> list[str]:
        return sorted(self._chunkers.keys())
