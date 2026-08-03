"""BM25 sparse retriever — a thin wrapper around a pluggable :class:`SparseIndex`.

The default backend is the pure-Python :class:`MemorySparseIndex` (Okapi BM25,
zero external dependencies, CJK-aware tokenizer).  Pass a custom ``index``
implementing the :class:`SparseIndex` protocol to plug in an external engine
such as Elasticsearch or meilisearch.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from raglan.retrievers.sparse import MemorySparseIndex, SparseIndex
from raglan.types import Filter, ScoredChunk


class BM25Retriever:
    """BM25 sparse retriever backed by a configurable sparse index.

    Suitable for corpora up to ~1 million documents on commodity hardware with
    the default in-memory backend.  For larger collections, pass a custom
    ``index`` backed by Elasticsearch or a similar engine.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter. Higher values give more
        weight to repeated terms within a document.
    b:
        Document-length normalisation parameter. ``0`` disables length
        normalisation; ``1`` applies full normalisation.
    tokenizer:
        Optional callable ``(text: str) -> list[str]``. When ``None``,
        the built-in Chinese-bigram / English-whitespace tokenizer is used.
    stopwords:
        Optional set of lowercase words to exclude from the index.
        When ``None`` a small built-in English stopword set is used.
    index:
        A :class:`SparseIndex` backend.  Defaults to :class:`MemorySparseIndex`.
    """

    name = "bm25"
    requires_embeddings = False

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
        stopwords: set[str] | None = None,
        index: SparseIndex | None = None,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer
        self._stopwords = stopwords
        self._index: SparseIndex = index or MemorySparseIndex(
            k1=k1, b=b, tokenizer=tokenizer, stopwords=stopwords
        )

    @staticmethod
    def _builtin_tokenizer(text: str) -> list[str]:
        """The default CJK-aware tokenizer (delegates to MemorySparseIndex)."""
        return MemorySparseIndex._builtin_tokenizer(text)

    # ------------------------------------------------------------------
    # Retriever protocol
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
    ) -> list[list[ScoredChunk]]:
        """Search for *top_k* chunks per query.

        *embeddings*, *filters*, and *timeout* are accepted for protocol
        compatibility.  *filters* is not supported by the default backend.
        """
        if filters:
            logging.getLogger(__name__).warning(
                "BM25Retriever does not support metadata filters — "
                "filters will be ignored. Use a dense retriever for "
                "filtered search."
            )
        results: list[list[ScoredChunk]] = []
        for query in queries:
            chunks = await self._index.search(query, top_k)
            for c in chunks:
                c.source = self.name
            results.append(chunks)
        return results

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        """Build the index from a stream of chunk batches."""
        await self._index.index(chunks)

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Incrementally add chunks to an existing index."""
        await self._index.add(chunks)

    async def remove(self, chunk_ids: list[str]) -> None:
        """Incrementally remove chunks from the index."""
        await self._index.remove(chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "k1": self._k1,
                "b": self._b,
            },
        }
