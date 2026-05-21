"""FastAPI application exposing /query and /health endpoints.

Run with:
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import AppConfig, load_config
from src.embeddings import EmbeddingService
from src.llm import LiteLLMClient
from src.retrieval import Retriever
from src.vectorstore import QdrantStore

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of chunks to retrieve.")
    use_llm: Optional[bool] = Field(
        None,
        description="If set, overrides the LLM-enabled flag from the configuration.",
    )


class RetrievedChunk(BaseModel):
    id: str
    score: float
    text: str
    source_file: str
    category: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    answer: Optional[str]
    used_llm: bool
    retrieved_chunks: List[RetrievedChunk]


class _AppState:
    """Holds long-lived components (loaded once at startup)."""

    cfg: AppConfig
    embedder: EmbeddingService
    store: QdrantStore
    retriever: Retriever
    llm: Optional[LiteLLMClient]


state = _AppState()
app = FastAPI(
    title="Engineering Knowledge RAG API",
    description="Semantic search over engineering documentation, code, configs and assembler.",
    version="0.1.0",
)


def _any_llm_key() -> bool:
    return any(
        os.getenv(k)
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHERAI_API_KEY",
        )
    )


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.getenv("RAG_CONFIG", "config/sources.yaml")
    logger.info("Loading config from %s", cfg_path)
    state.cfg = load_config(cfg_path)
    state.embedder = EmbeddingService(model_name=state.cfg.embedding.model)
    state.store = QdrantStore(
        path=state.cfg.vectorstore.path,
        collection=state.cfg.vectorstore.collection,
    )
    state.retriever = Retriever(state.embedder, state.store)
    state.llm = LiteLLMClient(model=state.cfg.llm.model) if state.cfg.llm.enabled else None
    logger.info("App started. LLM enabled=%s", state.cfg.llm.enabled)


@app.on_event("shutdown")
def _shutdown() -> None:
    try:
        state.store.close()
    except Exception:  # pragma: no cover
        pass


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        count = state.store.count()
    except Exception:
        count = -1
    return {
        "status": "ok",
        "collection": state.cfg.vectorstore.collection,
        "chunks_indexed": count,
        "llm_enabled": state.cfg.llm.enabled,
        "llm_model": state.cfg.llm.model,
        "embedding_model": state.cfg.embedding.model,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    top_k = req.top_k or state.cfg.retrieval.top_k
    hits = state.retriever.retrieve(req.question, top_k=top_k)

    chunks = [
        RetrievedChunk(
            id=h.id,
            score=h.score,
            text=h.text,
            source_file=h.source_file,
            category=h.category,
            metadata=h.metadata,
        )
        for h in hits
    ]

    # decide whether to run the LLM
    use_llm = state.cfg.llm.enabled if req.use_llm is None else req.use_llm
    if use_llm and not _any_llm_key():
        logger.warning("LLM requested but no API key set - returning retrieval only.")
        use_llm = False

    answer: Optional[str] = None
    if use_llm:
        llm = state.llm or LiteLLMClient(model=state.cfg.llm.model)
        try:
            answer = llm.generate_answer(req.question, hits)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM generation failed")
            raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    return QueryResponse(
        question=req.question,
        answer=answer,
        used_llm=bool(answer is not None),
        retrieved_chunks=chunks,
    )
