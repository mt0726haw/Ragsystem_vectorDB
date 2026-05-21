"""Configuration loading for the RAG system."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/sources.yaml")


@dataclass
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorStoreConfig:
    type: str = "qdrant"
    path: str = "./vectorstore/qdrant"
    collection: str = "engineering_knowledge"


@dataclass
class RetrievalConfig:
    top_k: int = 5


@dataclass
class LLMConfig:
    provider: str = "litellm"
    model: str = "openai/gpt-4o-mini"
    enabled: bool = False


@dataclass
class AppConfig:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load the YAML configuration into a typed AppConfig.

    Environment variables can override individual values:
      - QDRANT_PATH overrides vectorstore.path
      - LITELLM_MODEL overrides llm.model
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    embedding = EmbeddingConfig(**(raw.get("embedding") or {}))
    vectorstore = VectorStoreConfig(**(raw.get("vectorstore") or {}))
    retrieval = RetrievalConfig(**(raw.get("retrieval") or {}))
    llm = LLMConfig(**(raw.get("llm") or {}))

    # ENV overrides
    if env_path := os.getenv("QDRANT_PATH"):
        vectorstore.path = env_path
    if env_model := os.getenv("LITELLM_MODEL"):
        llm.model = env_model

    cfg = AppConfig(
        sources=raw.get("sources", []) or [],
        embedding=embedding,
        vectorstore=vectorstore,
        retrieval=retrieval,
        llm=llm,
    )

    logger.debug("Loaded config from %s: %s", cfg_path, cfg)
    return cfg
