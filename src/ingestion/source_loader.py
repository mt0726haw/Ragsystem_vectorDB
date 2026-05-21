"""Source loader: walks configured source directories and yields Documents."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from src.ingestion.parser_registry import ParserRegistry
from src.model import Document

logger = logging.getLogger(__name__)


class SourceLoader:
    """Loads documents from the source definitions in sources.yaml.

    Each source has a name, a base path, a category and an `include` list of
    glob patterns. Files are discovered recursively, and the matching parser
    is selected via the ParserRegistry.
    """

    def __init__(self, sources: List[Dict[str, Any]], parser_registry: ParserRegistry) -> None:
        self._sources = sources
        self._parsers = parser_registry

    def load(self) -> Iterator[Document]:
        for source in self._sources:
            yield from self._load_source(source)

    def _load_source(self, source: Dict[str, Any]) -> Iterator[Document]:
        name = source.get("name", "<unnamed>")
        category = source.get("category", "unknown")
        base = Path(source.get("path", ".")).resolve()
        includes: List[str] = source.get("include", ["*"])

        if not base.exists():
            logger.warning("Source %s: path %s does not exist - skipping", name, base)
            return

        logger.info("Scanning source %r at %s (category=%s)", name, base, category)
        files = sorted(self._iter_files(base, includes))
        logger.info("Source %r -> %d files", name, len(files))

        for file_path in files:
            parser = self._parsers.get(file_path)
            if parser is None:
                logger.debug("No parser for %s - skipping", file_path)
                continue
            try:
                doc = parser.parse(file_path, source_name=name, category=category)
                yield doc
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to parse %s: %s", file_path, exc)

    @staticmethod
    def _iter_files(base: Path, includes: Iterable[str]) -> Iterator[Path]:
        seen: set[Path] = set()
        for pattern in includes:
            for match in base.rglob(pattern):
                if match.is_file() and match not in seen:
                    seen.add(match)
                    yield match
