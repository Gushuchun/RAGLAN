"""Passthrough context builder — returns chunks as-is."""

from __future__ import annotations

from raglan.types import ScoredChunk, SearchResult


class PassthroughBuilder:
    """Returns chunks exactly as they were retrieved, with no expansion.

    This is the default context builder when the user does not supply a
    ``parent_chunk_loader``.  Results are wrapped in ``SearchResult``
    objects but content is not modified.
    """

    name = "passthrough"

    async def build(
        self,
        query: str,
        candidates: list[ScoredChunk],
        max_tokens: int = 6000,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=c.score,
                metadata=c.chunk_metadata,
                source=c.source,
            )
            for c in candidates
        ]
