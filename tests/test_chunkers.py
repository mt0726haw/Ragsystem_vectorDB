"""Unit tests for parsers, chunkers and registries.

These tests do *not* require sentence-transformers, qdrant-client or any
network access - so they run fast in CI and protect the chunking logic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.chunkers import AsmChunker, MarkdownChunker, PythonAstChunker, YamlChunker
from src.ingestion import ChunkerRegistry, ParserRegistry
from src.model import Document


@pytest.fixture
def tmp_md(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n"
        "Intro paragraph.\n\n"
        "## Section A\n"
        "Content A.\n\n"
        "## Section B\n"
        "Content B.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_py(tmp_path: Path) -> Path:
    p = tmp_path / "mod.py"
    p.write_text(
        '"""Module docstring."""\n'
        "X = 1\n\n"
        "def foo(x):\n"
        "    return x + 1\n\n"
        "class Bar:\n"
        "    def baz(self):\n"
        "        return 42\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "config.yml"
    p.write_text(
        "variables:\n"
        "  A: 1\n"
        "  B: 2\n"
        "stages:\n"
        "  - build\n"
        "  - deploy\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def tmp_asm(tmp_path: Path) -> Path:
    p = tmp_path / "boot.asm"
    p.write_text(
        "; header\n"
        "init:\n"
        "    mov sp, #0x100\n"
        "main_loop:\n"
        "    nop\n"
        "    b main_loop\n",
        encoding="utf-8",
    )
    return p


# ----- parsers / registry -----


def test_parser_registry_routes_by_extension(tmp_md, tmp_py, tmp_yaml, tmp_asm):
    reg = ParserRegistry()
    assert reg.get(tmp_md) is not None
    assert reg.get(tmp_py) is not None
    assert reg.get(tmp_yaml) is not None
    assert reg.get(tmp_asm) is not None
    assert reg.get(Path("foo.unknown")) is None


def test_chunker_registry_known_categories():
    reg = ChunkerRegistry()
    assert reg.get("documentation") is not None
    assert reg.get("python_code") is not None
    assert reg.get("yaml_config") is not None
    assert reg.get("assembler") is not None
    assert reg.get("nope") is None


# ----- chunkers -----


def _doc(path: Path, category: str) -> Document:
    return Document(
        source_name="test",
        category=category,
        file_path=path,
        file_type=path.suffix.lstrip("."),
        content=path.read_text(encoding="utf-8"),
    )


def test_markdown_chunker_splits_by_heading(tmp_md):
    doc = _doc(tmp_md, "documentation")
    chunks = MarkdownChunker().chunk(doc)
    headings = [c.metadata["heading"] for c in chunks]
    assert "Title" in headings
    assert "Section A" in headings
    assert "Section B" in headings


def test_python_ast_chunker_emits_function_class_method(tmp_py):
    doc = _doc(tmp_py, "python_code")
    chunks = PythonAstChunker().chunk(doc)
    kinds = {(c.metadata.get("node_type"), c.metadata.get("function"), c.metadata.get("class")) for c in chunks}
    # Expect function 'foo', class 'Bar', method 'baz' under Bar
    assert any(t[1] == "foo" for t in kinds)
    assert any(t[2] == "Bar" and t[0] == "ClassDef" for t in kinds)
    assert any(t[1] == "baz" and t[2] == "Bar" for t in kinds)


def test_yaml_chunker_one_per_top_level_key(tmp_yaml):
    doc = _doc(tmp_yaml, "yaml_config")
    chunks = YamlChunker().chunk(doc)
    keys = sorted(c.metadata["top_level_key"] for c in chunks)
    assert keys == ["stages", "variables"]


def test_asm_chunker_splits_by_label(tmp_asm):
    doc = _doc(tmp_asm, "assembler")
    chunks = AsmChunker().chunk(doc)
    labels = [c.metadata["label"] for c in chunks]
    assert "init" in labels
    assert "main_loop" in labels


def test_chunk_id_is_deterministic(tmp_md):
    doc = _doc(tmp_md, "documentation")
    a = MarkdownChunker().chunk(doc)
    b = MarkdownChunker().chunk(doc)
    assert [c.id for c in a] == [c.id for c in b]
