"""Query CLI.

Usage:
    python query.py --question "How does the demo pipeline work?" --top-k 5

Without an enabled LLM, this prints the top-k retrieved chunks with their
source, category, metadata and a text excerpt. When the LLM is enabled in
the config (and an API key is provided), it additionally generates a
grounded answer via LiteLLM.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from typing import List

from dotenv import load_dotenv

from config import AppConfig, load_config
from src.embeddings import EmbeddingService
from src.llm import LiteLLMClient
from src.retrieval import Retriever
from src.vectorstore import QdrantStore, SearchHit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a semantic query against the vector DB.")
    parser.add_argument("--question", "-q", required=True, help="Natural-language question.")
    parser.add_argument(
        "--config",
        default="config/sources.yaml",
        help="Path to the sources YAML configuration (default: config/sources.yaml)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=None, help="Number of chunks to retrieve."
    )
    parser.add_argument(
        "--llm",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Whether to invoke the LLM. 'auto' (default) follows the config, "
            "'on' forces it, 'off' disables it."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_hits(hits: List[SearchHit]) -> None:
    if not hits:
        print("No matching chunks found.")
        return
    print()
    print("=" * 70)
    print(f"TOP {len(hits)} RETRIEVED CHUNKS")
    print("=" * 70)
    for i, hit in enumerate(hits, start=1):
        print(f"\n[{i}] score={hit.score:.4f}  category={hit.category}")
        print(f"    source : {hit.source_file}")
        if hit.metadata:
            meta = ", ".join(f"{k}={v}" for k, v in hit.metadata.items() if v is not None)
            if meta:
                print(f"    meta   : {meta}")
        excerpt = textwrap.shorten(hit.text.replace("\n", " "), width=400, placeholder=" ...")
        print(f"    text   : {excerpt}")
    print("=" * 70)


def llm_enabled(cfg: AppConfig, override: str) -> bool:
    if override == "off":
        return False
    if override == "on":
        return True
    return bool(cfg.llm.enabled)


def has_any_llm_key() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHERAI_API_KEY",
        )
    )


def main() -> int:
    load_dotenv()
    args = parse_args()
    configure_logging(args.verbose)
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    top_k = args.top_k or cfg.retrieval.top_k

    embedder = EmbeddingService(model_name=cfg.embedding.model)
    store = QdrantStore(path=cfg.vectorstore.path, collection=cfg.vectorstore.collection)
    retriever = Retriever(embedder, store)

    hits = retriever.retrieve(args.question, top_k=top_k)
    print_hits(hits)

    use_llm = llm_enabled(cfg, args.llm)
    if use_llm and not has_any_llm_key():
        print(
            "\n[!] LLM enabled but no API key found in environment - "
            "set OPENAI_API_KEY / ANTHROPIC_API_KEY / ... in .env.\n"
            "    Falling back to retrieval-only output.",
            file=sys.stderr,
        )
        use_llm = False

    if use_llm:
        print("\nGenerating LLM answer ...\n")
        client = LiteLLMClient(model=cfg.llm.model)
        try:
            answer = client.generate_answer(args.question, hits)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[!] LLM call failed: {exc}", file=sys.stderr)
            store.close()
            return 1
        print("=" * 70)
        print("LLM ANSWER")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    else:
        print(
            "\n[i] LLM is disabled - showing retrieval context only. "
            "Set llm.enabled=true in config + an API key in .env to enable answers."
        )

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
