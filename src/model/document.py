"""Document model representing a single source file before chunking."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class Document:
    """A raw document loaded from a source.

    Attributes:
        source_name: Logical name of the source (from sources.yaml).
        category: Content category (documentation, python_code, yaml_config, assembler, ...).
        file_path: Absolute path to the file.
        file_type: File extension (without dot, lower-case).
        content: Raw textual content of the file.
        metadata: Additional metadata (e.g. mtime, size).
    """

    source_name: str
    category: str
    file_path: Path
    file_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def file_name(self) -> str:
        return self.file_path.name
