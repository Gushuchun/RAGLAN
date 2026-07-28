"""OpenAI embedder.

Requires ``pip install raglan-retrieval[openai]`` (or ``openai``).
"""

from __future__ import annotations

from typing import Any, ClassVar


class OpenAIEmbedder:
    """Generates embeddings via the OpenAI Embeddings API.

    Parameters
    ----------
    model:
        Model name (``text-embedding-3-small``, ``text-embedding-3-large``).
    batch_size:
        Maximum texts per API call.  OpenAI allows up to 2048.
    base_url:
        Optional base URL for OpenAI-compatible proxies.
    api_key:
        Optional API key.  If ``None`` the ``OPENAI_API_KEY`` env var is used.
    """

    name = "openai_embedder"
    # Class-level default so @runtime_checkable Embedder Protocol passes
    # isinstance checks even before __init__ runs.  Per-instance value is
    # set in __init__ based on the chosen model.
    dimension = 1536

    _DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        batch_size: int = 100,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None
        self.dimension = self._DIMENSIONS.get(model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])

        return all_embeddings

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from raglan._lazy import _import_module

        _import_module("openai", hint="pip install raglan-retrieval[openai]")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    async def close(self) -> None:
        """Close the underlying OpenAI HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model": self._model,
                "batch_size": self._batch_size,
                "base_url": self._base_url,
                "api_key": "<redacted>" if self._api_key else None,
            },
        }
