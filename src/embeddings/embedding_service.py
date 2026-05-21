"""Embedding service backed by sentence-transformers."""
from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a sentence-transformers model for batched encoding.

    The model is loaded lazily so importing this module is cheap and the
    heavy ML stack isn't pulled in unless embeddings are actually needed.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dimension: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading embedding model %s ...", self._model_name)
            from sentence_transformers import SentenceTransformer  # local import

            self._model = SentenceTransformer(self._model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info("Loaded embedding model %s (dim=%d)", self._model_name, self._dimension)

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode a sequence of texts and return list of float vectors."""
        self._ensure_loaded()
        if not texts:
            return []
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in vectors]

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]
