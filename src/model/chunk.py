"""Chunk model representing a single embeddable text chunk."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """A single chunk of text ready for embedding & storage.

    Attributes:
        id: Stable UUID for this chunk (derived from source_file + content hash).
        text: The chunk text content.
        source_file: Path of the originating file (string for serialization).
        category: Content category propagated from the Document.
        metadata: Chunker-specific metadata (heading, function, label, ...).
    """

    text: str
    source_file: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self._make_id()

    def _make_id(self) -> str:
        """Generate a deterministic UUIDv5 id based on file + content hash.

        This keeps re-ingestion idempotent: same chunk -> same id -> upsert.
        """
        digest = hashlib.sha256(
            f"{self.source_file}:{self.text}".encode("utf-8")
        ).hexdigest()
        return str(uuid.UUID(digest[:32]))

    def payload(self) -> Dict[str, Any]:
        """Build the Qdrant payload (everything except the vector)."""
        return {
            "text": self.text,
            "source_file": self.source_file,
            "category": self.category,
            **self.metadata,
        }
