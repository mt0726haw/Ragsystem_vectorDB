"""End-to-end smoke test using a deterministic stub embedder.

Verifies the full ingest -> Qdrant upsert -> retrieve loop without needing
network access to download a real embedding model.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path
from typing import List, Sequence

import pytest

# Make sure the repo root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


pytest.importorskip("qdrant_client")


class StubEmbedder:
    """Deterministic, hash-based fake embedder (no network, no torch)."""

    dimension = 32
    model_name = "stub"

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._hash_to_vec(t) for t in texts]

    def encode_one(self, text: str) -> List[float]:
        return self._hash_to_vec(text)

    def _hash_to_vec(self, text: str) -> List[float]:
        # Build a length-`dimension` pseudo-random unit vector from text hash.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat & slice to required length
        raw = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
        vec = [(b - 128) / 128.0 for b in raw]
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


def test_end_to_end_ingest_and_retrieve(tmp_path: Path) -> None:
    from src.ingestion import ChunkerRegistry, ParserRegistry, SourceLoader
    from src.vectorstore import QdrantStore

    # Build a tiny synthetic source tree
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "intro.md").write_text(
        "# Pipeline Overview\n"
        "The pipeline ingests data and produces aggregated results.\n\n"
        "## Stages\n"
        "Normalize then aggregate.\n",
        encoding="utf-8",
    )
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "mod.py").write_text(
        "def process_data(records):\n"
        "    return [r for r in records if r]\n",
        encoding="utf-8",
    )

    sources = [
        {
            "name": "docs",
            "path": str(docs_dir),
            "category": "documentation",
            "include": ["*.md"],
        },
        {
            "name": "code",
            "path": str(code_dir),
            "category": "python_code",
            "include": ["*.py"],
        },
    ]

    parser_reg = ParserRegistry()
    chunker_reg = ChunkerRegistry()
    loader = SourceLoader(sources, parser_reg)

    documents = list(loader.load())
    assert len(documents) == 2

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunker_reg.get(doc.category).chunk(doc))
    assert len(all_chunks) >= 3

    embedder = StubEmbedder()
    vectors = embedder.encode([c.text for c in all_chunks])

    store_path = tmp_path / "vstore"
    store = QdrantStore(path=store_path, collection="test_kb")
    try:
        store.ensure_collection(vector_size=embedder.dimension)
        inserted = store.upsert(all_chunks, vectors)
        assert inserted == len(all_chunks)
        assert store.count() == len(all_chunks)

        # Retrieve: we can't assert semantic relevance with a hash-based
        # stub embedder, but we *can* assert the round-trip works:
        # - querying returns the requested number of hits
        # - payload (text, source_file, category, metadata) survives storage
        # - querying the *exact* chunk text yields a perfect score 1.0
        query_vec = embedder.encode_one("Pipeline Overview The pipeline ingests data")
        hits = store.search(query_vec, top_k=3)
        assert hits, "expected at least one hit"
        assert all(h.text for h in hits)
        assert all(h.source_file for h in hits)
        assert {h.category for h in hits} <= {"documentation", "python_code"}

        # Querying with the exact chunk text -> perfect cosine similarity.
        exact_chunk = all_chunks[0]
        exact_hits = store.search(embedder.encode_one(exact_chunk.text), top_k=1)
        assert exact_hits[0].score > 0.99
        assert exact_hits[0].id == exact_chunk.id
    finally:
        store.close()
        shutil.rmtree(store_path, ignore_errors=True)


def test_idempotent_reingest(tmp_path: Path) -> None:
    """Re-running ingestion on the same content should not duplicate points."""
    from src.chunkers import MarkdownChunker
    from src.model import Document
    from src.vectorstore import QdrantStore

    md = tmp_path / "x.md"
    md.write_text("# H1\nbody\n## H2\nbody2\n", encoding="utf-8")
    doc = Document(
        source_name="s",
        category="documentation",
        file_path=md,
        file_type="md",
        content=md.read_text(),
    )
    chunks = MarkdownChunker().chunk(doc)
    embedder = StubEmbedder()
    vectors = embedder.encode([c.text for c in chunks])

    store_path = tmp_path / "v"
    store = QdrantStore(path=store_path, collection="kb")
    try:
        store.ensure_collection(vector_size=embedder.dimension)
        store.upsert(chunks, vectors)
        first = store.count()
        # Re-ingest exact same chunks
        store.upsert(chunks, vectors)
        second = store.count()
        assert first == second, "deterministic IDs should keep counts stable"
    finally:
        store.close()
