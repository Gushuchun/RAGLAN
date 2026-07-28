"""Alibaba DashScope embedder — Tongyi embedding models.

Requires ``pip install raglan-retrieval[dashscope]`` (or ``dashscope``).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from raglan.exceptions import EmbedderError


class DashScopeEmbedder:
    """Generates embeddings via Alibaba DashScope (Tongyi) API.

    Parameters
    ----------
    model:
        Model name (``text-embedding-v3``, ``text-embedding-v2``,
        ``text-embedding-async-v2``).
    batch_size:
        Maximum texts per API call.  DashScope supports up to 25 for v3.
    api_key:
        Optional API key.  If ``None`` the ``DASHSCOPE_API_KEY`` env var
        is used.
    """

    name = "dashscope_embedder"

    _DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-v3": 1024,
        "text-embedding-v2": 1536,
        "text-embedding-async-v2": 1536,
        "text-embedding-v1": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-v3",
        *,
        batch_size: int = 10,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._api_key = api_key
        self.dimension = self._DIMENSIONS.get(model, 1024)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from raglan._lazy import _import_module

        _import_module("dashscope", hint="pip install raglan-retrieval[dashscope]")
        import dashscope

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # Pass API key per-request rather than mutating global state
            call_kwargs: dict[str, Any] = {
                "model": self._model,
                "input": batch,
            }
            if self._api_key is not None:
                call_kwargs["api_key"] = self._api_key

            resp = await asyncio.to_thread(
                dashscope.TextEmbedding.call,
                **call_kwargs,
            )
            if resp.status_code != 200:
                raise EmbedderError(f"DashScope embedding failed: {resp.code} - {resp.message}")
            all_embeddings.extend([emb["embedding"] for emb in resp.output["embeddings"]])

        return all_embeddings

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model": self._model,
                "batch_size": self._batch_size,
                "api_key": "<redacted>" if self._api_key else None,
            },
        }
