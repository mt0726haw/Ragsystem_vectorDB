"""YAML chunker - one chunk per top-level key."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import yaml

from src.model import Chunk, Document

logger = logging.getLogger(__name__)


class YamlChunker:
    """Chunks YAML documents by their top-level keys.

    For each top-level mapping key, a chunk is produced containing the
    YAML-serialized sub-tree. If the YAML root is a list or scalar, a
    single chunk is produced.
    """

    def chunk(self, document: Document) -> List[Chunk]:
        try:
            parsed = yaml.safe_load(document.content)
        except yaml.YAMLError as exc:
            logger.warning("YAML parse error in %s: %s - using single chunk", document.file_name, exc)
            return self._single(document)

        if not isinstance(parsed, dict):
            return self._single(document)

        chunks: List[Chunk] = []
        for top_key, sub in parsed.items():
            try:
                serialized = yaml.safe_dump(
                    {top_key: sub}, sort_keys=False, default_flow_style=False
                ).strip()
            except yaml.YAMLError:
                serialized = f"{top_key}: <unserializable>"

            chunks.append(
                Chunk(
                    text=serialized,
                    source_file=str(document.file_path),
                    category=document.category,
                    metadata={
                        "top_level_key": str(top_key),
                        "yaml_path": f"$.{top_key}",
                        "source_file": document.file_name,
                    },
                )
            )

        logger.debug("YAML chunker produced %d chunks for %s", len(chunks), document.file_name)
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
                    "top_level_key": "(root)",
                    "yaml_path": "$",
                    "source_file": document.file_name,
                },
            )
        ]
