"""High-level retriever combining embedding + vector store search."""
from __future__ import annotations

import logging
from typing import List

from src.embeddings import EmbeddingService
from src.vectorstore import QdrantStore, SearchHit

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: EmbeddingService, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, top_k: int = 5) -> List[SearchHit]:
        logger.debug("Retrieving top-%d for question: %s", top_k, question)
        vector = self._embedder.encode_one(question)
        hits = self._store.search(vector, top_k=top_k)
        logger.debug("Retrieved %d hits", len(hits))
        return hits
