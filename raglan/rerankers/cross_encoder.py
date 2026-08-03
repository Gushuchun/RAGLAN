"""Cross-Encoder reranker backed by sentence-transformers.

Requires ``pip install raglan-retrieval[huggingface]`` (or ``sentence-transformers``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from raglan.types import ScoredChunk


class CrossEncoderReranker:
    """Re-ranks candidates using a sentence-transformers Cross-Encoder model.

    Parameters
    ----------
    model_name:
        Any HuggingFace Cross-Encoder model.  The default
        ``ms-marco-TinyBERT-L2-v2`` is ~20 MB and runs comfortably on CPU.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    batch_size:
        How many (query, document) pairs to score in one call.
    min_score:
        Candidates scoring below this threshold are discarded.
    input_builder:
        Optional callback ``(query: str, doc: str) -> str`` to customise
        the input format for different Cross-Encoder models.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L2-v2",
        *,
        device: str | None = None,
        batch_size: int = 8,
        min_score: float = 0.0,
        input_builder: Callable[[str, str], str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._min_score = min_score
        self._input_builder = input_builder or self._default_input
        self._model: Any = None

    async def warm_up(self) -> None:
        """Pre-load the Cross-Encoder model so the first ``rerank()`` is fast.

        Call during application startup (optionally via ``Raglan.warm_up()``)
        to avoid the model-download/load stall on the first request.
        """
        await asyncio.to_thread(self._get_model)

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        model = self._get_model()
        threshold = min_score if min_score > 0 else self._min_score

        pairs = [self._input_builder(query, c.content) for c in candidates]

        scores = await asyncio.to_thread(
            model.predict,
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        ranked: list[ScoredChunk] = []
        for chunk, score in zip(candidates, scores, strict=False):
            s = float(score)
            if s >= threshold:
                chunk.score = s
                chunk.source = f"{chunk.source}→cross_encoder"
                ranked.append(chunk)

        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model_name": self._model_name,
                "device": self._device,
                "batch_size": self._batch_size,
                "min_score": self._min_score,
            },
        }

    @staticmethod
    def _default_input(query: str, doc: str) -> str:
        return f"Query: {query} Document: {doc}"

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        from raglan._lazy import _import_module

        _import_module("sentence_transformers", hint="pip install raglan-retrieval[huggingface]")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
        )
        return self._model
