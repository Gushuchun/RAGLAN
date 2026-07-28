"""In-memory brute-force retriever — for testing and tiny datasets."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from typing import Any

from raglan.types import Filter, ScoredChunk


class MemoryRetriever:
    """Brute-force cosine-similarity retriever backed by a Python list.

    Every query is compared against every chunk via cosine similarity.
    This is O(N·D) per query and should only be used for testing or
    datasets smaller than ~10 000 chunks.

    All mutations are protected by an ``asyncio.Lock`` for safe concurrent
    use.

    Parameters
    ----------
    chunks:
        Initial set of ``(chunk_id, content, embedding, metadata)`` tuples.
    """

    name = "memory"
    requires_embeddings = True

    def __init__(
        self,
        chunks: list[tuple[str, str, list[float], dict[str, Any] | None]] | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        # (chunk_id, content, embedding, metadata)
        self._chunks: list[tuple[str, str, list[float], dict[str, Any]]] = []
        if chunks:
            for cid, content, emb, meta in chunks:
                self._chunks.append((cid, content, emb, meta or {}))

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
        async with self._lock:
            results: list[list[ScoredChunk]] = []
            for emb in embeddings:
                scored = []
                for cid, content, chunk_emb, meta in self._chunks:
                    sim = _cosine(emb, chunk_emb)
                    scored.append(
                        ScoredChunk(
                            chunk_id=cid,
                            content=content,
                            score=sim,
                            chunk_metadata=meta,
                            source=self.name,
                        )
                    )
                scored.sort(key=lambda c: c.score, reverse=True)
                results.append(scored[:top_k])
            return results

    @staticmethod
    def _unpack_item(
        item: tuple[Any, ...],
    ) -> tuple[str, str, list[float], dict[str, Any]]:
        """Safely unpack a chunk tuple — supports 2-, 3-, and 4-element formats.

        Formats accepted:
        - ``(chunk_id, content)``
        - ``(chunk_id, content, metadata)``
        - ``(chunk_id, content, metadata, embedding)``
        """
        cid = str(item[0])
        content = str(item[1])
        meta: dict[str, Any] | None = item[2] if len(item) > 2 else None
        emb: list[float] = []
        if len(item) > 3 and item[3] is not None:
            emb = list(item[3])
        return cid, content, emb, meta or {}

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        async with self._lock:
            self._chunks.clear()
            async for batch in chunks:
                for item in batch:
                    self._chunks.append(self._unpack_item(item))

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        async with self._lock:
            for item in chunks:
                self._chunks.append(self._unpack_item(item))

    async def remove(self, chunk_ids: list[str]) -> None:
        async with self._lock:
            ids = set(chunk_ids)
            self._chunks = [(cid, c, e, m) for cid, c, e, m in self._chunks if cid not in ids]

    # ------------------------------------------------------------------
    # Convenience — load with embeddings for direct (non-pipeline) use
    # ------------------------------------------------------------------

    def load_embedded(
        self,
        items: list[tuple[str, str, list[float], dict[str, Any] | None]],
    ) -> None:
        """Directly load chunks with pre-computed embeddings."""
        self._chunks = [(cid, c, e, m or {}) for cid, c, e, m in items]

    def to_dict(self) -> dict[str, Any]:
        # MemoryRetriever is not serialisable — chunk data lives in memory.
        # The returned dict is only useful for identifying the component type.
        return {"type": self.name, "params": {}}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
