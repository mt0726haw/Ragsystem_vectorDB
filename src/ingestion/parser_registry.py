"""Parser registry: maps file extensions to parser instances."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from src.model import Document
from src.parsers import AsmParser, MarkdownParser, PythonParser, YamlParser

logger = logging.getLogger(__name__)


class Parser(Protocol):
    extensions: tuple

    def parse(self, file_path: Path, source_name: str, category: str) -> Document: ...


class ParserRegistry:
    """Selects the right parser based on the file extension."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Any] = {}
        # Default registrations
        self.register(MarkdownParser())
        self.register(PythonParser())
        self.register(YamlParser())
        self.register(AsmParser())

    def register(self, parser: Any) -> None:
        for ext in parser.extensions:
            key = ext.lower().lstrip(".")
            self._parsers[key] = parser
            logger.debug("Registered parser for .%s -> %s", key, type(parser).__name__)

    def get(self, file_path: Path) -> Optional[Any]:
        ext = file_path.suffix.lower().lstrip(".")
        return self._parsers.get(ext)

    def supported_extensions(self) -> list[str]:
        return sorted(self._parsers.keys())
