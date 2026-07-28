"""Cohere Rerank API reranker.

Requires ``pip install raglan-retrieval[cohere]`` (or ``cohere``).
"""

from __future__ import annotations

from typing import Any

from raglan.types import ScoredChunk


class CohereReranker:
    """Re-ranks candidates using the Cohere Rerank API.

    Parameters
    ----------
    model:
        Cohere rerank model (``rerank-v3.5``, ``rerank-english-v3.0``,
        ``rerank-multilingual-v3.0``).
    top_n:
        Maximum number of results to return.
    min_score:
        Candidates scoring below this threshold are discarded.
    api_key:
        Optional API key.  If ``None`` the ``CO_API_KEY`` env var is used.
    base_url:
        Optional custom API base URL.
    """

    name = "cohere_reranker"

    def __init__(
        self,
        model: str = "rerank-v3.5",
        *,
        top_n: int | None = None,
        min_score: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._top_n = top_n
        self._min_score = min_score
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        client = self._get_client()
        threshold = min_score if min_score > 0 else self._min_score

        documents = [c.content for c in candidates]
        resp = await client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=self._top_n or top_k,
        )

        ranked: list[ScoredChunk] = []
        for r in resp.results:
            score = float(r.relevance_score)
            if score >= threshold:
                chunk = candidates[r.index]
                chunk.score = score
                chunk.source = f"{chunk.source}→cohere"
                ranked.append(chunk)

        return ranked[:top_k]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from raglan._lazy import _import_module

        _import_module("cohere", hint="pip install raglan-retrieval[cohere]")
        import cohere

        client_kwargs: dict[str, Any] = {}
        if self._api_key is not None:
            client_kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            client_kwargs["base_url"] = self._base_url
        self._client = cohere.AsyncClientV2(**client_kwargs)
        return self._client

    async def close(self) -> None:
        """Close the underlying Cohere HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model": self._model,
                "top_n": self._top_n,
                "min_score": self._min_score,
                "base_url": self._base_url,
            },
        }
