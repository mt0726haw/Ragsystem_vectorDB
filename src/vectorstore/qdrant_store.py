"""Qdrant vector store wrapper (local mode)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.model import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    id: str
    score: float
    text: str
    source_file: str
    category: str
    metadata: Dict[str, Any]


class QdrantStore:
    """Thin wrapper around qdrant-client in local-mode.

    Persists data under `path` (a local directory). Creates the collection
    on demand and supports idempotent upserts (chunk ids are deterministic).
    """

    def __init__(
        self,
        path: str | Path = "./vectorstore/qdrant",
        collection: str = "engineering_knowledge",
        vector_size: Optional[int] = None,
        distance: str = "Cosine",
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection = collection
        self._distance = distance
        self._client = None
        self._vector_size = vector_size

    # ----- client lifecycle -----

    def _ensure_client(self) -> None:
        if self._client is None:
            from qdrant_client import QdrantClient  # local import

            logger.info("Opening Qdrant local store at %s", self._path)
            self._client = QdrantClient(path=str(self._path))

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not exist yet."""
        from qdrant_client.http import models as qm  # local import

        self._ensure_client()
        assert self._client is not None
        self._vector_size = vector_size

        collections = {c.name for c in self._client.get_collections().collections}
        if self._collection in collections:
            logger.debug("Collection %s already exists", self._collection)
            return

        distance_enum = getattr(qm.Distance, self._distance.upper(), qm.Distance.COSINE)
        logger.info(
            "Creating collection %s (size=%d, distance=%s)",
            self._collection,
            vector_size,
            distance_enum,
        )
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=distance_enum),
        )

    # ----- write -----

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[List[float]]) -> int:
        """Insert (or update) chunks together with their embedding vectors."""
        from qdrant_client.http import models as qm  # local import

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return 0

        self._ensure_client()
        assert self._client is not None

        points = [
            qm.PointStruct(id=chunk.id, vector=vec, payload=chunk.payload())
            for chunk, vec in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted %d points into %s", len(points), self._collection)
        return len(points)

    # ----- read -----

    def search(self, vector: List[float], top_k: int = 5) -> List[SearchHit]:
        self._ensure_client()
        assert self._client is not None

        # query_points is the current API; fall back to legacy `.search` if unavailable.
        try:
            result = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            points = result.points
        except AttributeError:  # pragma: no cover - older qdrant-client
            points = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )

        hits: List[SearchHit] = []
        for p in points:
            payload = dict(p.payload or {})
            hits.append(
                SearchHit(
                    id=str(p.id),
                    score=float(p.score),
                    text=payload.pop("text", ""),
                    source_file=payload.pop("source_file", ""),
                    category=payload.pop("category", ""),
                    metadata=payload,
                )
            )
        return hits

    # ----- introspection -----

    def count(self) -> int:
        self._ensure_client()
        assert self._client is not None
        try:
            return int(self._client.count(self._collection, exact=True).count)
        except Exception:  # pragma: no cover
            return 0

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None
