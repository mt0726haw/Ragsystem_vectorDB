"""Ingestion CLI.

Usage:
    python ingest.py --config config/sources.yaml

Loads sources, parses files, chunks them, embeds chunks and persists them
into the configured Qdrant collection. Re-running the script is idempotent
because chunk ids are deterministic.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from config import AppConfig, load_config
from src.embeddings import EmbeddingService
from src.ingestion import ChunkerRegistry, ParserRegistry, SourceLoader
from src.model import Chunk
from src.vectorstore import QdrantStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest configured sources into the vector DB.")
    parser.add_argument(
        "--config",
        default="config/sources.yaml",
        help="Path to the sources YAML configuration (default: config/sources.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_ingest(cfg: AppConfig) -> Dict[str, object]:
    logger = logging.getLogger("ingest")
    logger.info("Starting ingestion ...")

    parser_registry = ParserRegistry()
    chunker_registry = ChunkerRegistry()
    loader = SourceLoader(cfg.sources, parser_registry)

    documents = list(loader.load())
    logger.info("Loaded %d documents", len(documents))

    all_chunks: List[Chunk] = []
    per_category = Counter()
    per_source: Dict[str, int] = {}

    for doc in documents:
        chunker = chunker_registry.get(doc.category)
        if chunker is None:
            logger.warning("No chunker for category %r - skipping %s", doc.category, doc.file_path)
            continue
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        per_category[doc.category] += len(chunks)
        per_source[doc.source_name] = per_source.get(doc.source_name, 0) + len(chunks)

    logger.info("Produced %d chunks total", len(all_chunks))

    if not all_chunks:
        logger.warning("No chunks were produced - nothing to embed.")
        return {
            "files": len(documents),
            "chunks": 0,
            "per_category": dict(per_category),
            "per_source": per_source,
            "collection": cfg.vectorstore.collection,
        }

    embedder = EmbeddingService(model_name=cfg.embedding.model)
    logger.info("Embedding %d chunks with %s ...", len(all_chunks), cfg.embedding.model)
    vectors = embedder.encode([c.text for c in all_chunks])

    store = QdrantStore(
        path=cfg.vectorstore.path,
        collection=cfg.vectorstore.collection,
    )
    store.ensure_collection(vector_size=embedder.dimension)
    inserted = store.upsert(all_chunks, vectors)
    total = store.count()
    store.close()

    return {
        "files": len(documents),
        "chunks": len(all_chunks),
        "inserted": inserted,
        "collection_total": total,
        "per_category": dict(per_category),
        "per_source": per_source,
        "collection": cfg.vectorstore.collection,
        "vector_path": str(Path(cfg.vectorstore.path).resolve()),
    }


def print_summary(stats: Dict[str, object]) -> None:
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Files processed       : {stats['files']}")
    print(f"Chunks produced       : {stats['chunks']}")
    if "inserted" in stats:
        print(f"Chunks upserted       : {stats['inserted']}")
        print(f"Collection total      : {stats['collection_total']}")
    print(f"Collection            : {stats['collection']}")
    if "vector_path" in stats:
        print(f"Vector store location : {stats['vector_path']}")
    per_cat = stats.get("per_category", {}) or {}
    if per_cat:
        print("\nChunks per category:")
        for cat, n in sorted(per_cat.items()):
            print(f"  - {cat:<20} {n}")
    per_src = stats.get("per_source", {}) or {}
    if per_src:
        print("\nChunks per source:")
        for src, n in sorted(per_src.items()):
            print(f"  - {src:<20} {n}")
    print("=" * 60)


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.verbose)
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    stats = run_ingest(cfg)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
