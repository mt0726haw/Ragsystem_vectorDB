"""LiteLLM-based LLM client.

LiteLLM provides an OpenAI-compatible interface for 100+ LLM providers, so
swapping the model later only requires changing `model` and the corresponding
API key in the environment. By default the LLM is *disabled* - the system
returns retrieval context only, which means it is usable without any API key.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from src.vectorstore import SearchHit

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an engineering knowledge assistant. Answer the user question "
    "ONLY using the provided context. If the context does not contain enough "
    "information to answer, reply exactly: "
    "\"Die Wissensbasis enthält dazu keine ausreichende Antwort.\"\n"
    "Always cite the file names you used in square brackets at the end."
)


def build_context_block(chunks: Sequence[SearchHit]) -> str:
    """Render retrieved chunks into a single prompt-friendly context block."""
    lines: List[str] = []
    for i, hit in enumerate(chunks, start=1):
        meta = ", ".join(f"{k}={v}" for k, v in hit.metadata.items() if v is not None)
        lines.append(
            f"[{i}] file={hit.source_file} category={hit.category} score={hit.score:.3f}"
        )
        if meta:
            lines.append(f"    meta: {meta}")
        lines.append(hit.text)
        lines.append("---")
    return "\n".join(lines)


class LiteLLMClient:
    """Thin wrapper around `litellm.completion`."""

    def __init__(self, model: str = "openai/gpt-4o-mini", temperature: float = 0.1) -> None:
        self._model = model
        self._temperature = temperature

    def generate_answer(self, question: str, retrieved_chunks: Sequence[SearchHit]) -> str:
        try:
            import litellm  # local import - heavy
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "litellm is not installed - run `pip install litellm` first"
            ) from exc

        context = build_context_block(retrieved_chunks)
        user_msg = (
            f"Question:\n{question}\n\n"
            f"Context (top-{len(retrieved_chunks)} retrieved chunks):\n{context}"
        )

        logger.info("Calling LLM model=%s ...", self._model)
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self._temperature,
        )
        # LiteLLM mirrors the OpenAI response shape.
        return response["choices"][0]["message"]["content"]
