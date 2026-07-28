"""HuggingFace embedder — local models, no API calls.

Requires ``pip install raglan-retrieval[huggingface]`` (or ``sentence-transformers``).
"""

from __future__ import annotations

import asyncio
from typing import Any


class HuggingFaceEmbedder:
    """Generates embeddings with a local sentence-transformers model.

    No network calls after the initial model download.  Suitable for
    offline / air-gapped deployments and privacy-sensitive workloads.

    Embedding calls are serialised via an internal ``asyncio.Lock`` to
    prevent concurrent ``model.encode`` calls which SentenceTransformer
    does not support natively.

    Parameters
    ----------
    model_name:
        Any SentenceTransformer-compatible model.
    device:
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    normalize:
        Whether to L2-normalise output embeddings.
    """

    name = "huggingface_embedder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model: Any = None
        self._embed_lock = asyncio.Lock()
        # Determined lazily from model config on first load
        self.dimension = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._get_model()
        async with self._embed_lock:
            embeddings = await asyncio.to_thread(
                model.encode,
                texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        if self.dimension == 0 and len(embeddings) > 0:
            self.dimension = embeddings.shape[1]
        return [e.tolist() for e in embeddings]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        from raglan._lazy import _import_module

        _import_module("sentence_transformers", hint="pip install raglan-retrieval[huggingface]")
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )
        # Eagerly determine dimension from model config
        if self.dimension == 0:
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self.dimension = dim
        return self._model

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model_name": self._model_name,
                "device": self._device,
                "normalize": self._normalize,
            },
        }
