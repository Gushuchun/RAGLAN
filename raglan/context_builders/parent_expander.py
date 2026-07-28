"""Parent-chunk context expansion.

Loads full parent chunks for each candidate and greedily packs them
into the final context window up to *max_tokens*.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from raglan.types import ScoredChunk, SearchResult

ParentChunkLoader = Callable[[list[str]], Awaitable[dict[str, str]]]


class ParentExpander:
    """Expands matched child chunks into their full parent-document context.

    Parameters
    ----------
    loader:
        Async callable ``(chunk_ids: list[str]) -> dict[str, str]`` that
        maps child-chunk IDs to full parent-chunk content.
    max_tokens:
        Hard cap on total tokens in the expanded context.  Exceeding
        chunks are truncated (with a ``"..."`` suffix).
    """

    name = "parent_expander"

    def __init__(
        self,
        loader: ParentChunkLoader,
        *,
        max_tokens: int = 6000,
    ) -> None:
        self._loader = loader
        self._max_tokens = max_tokens

    async def build(
        self,
        query: str,
        candidates: list[ScoredChunk],
        max_tokens: int = -1,
    ) -> list[SearchResult]:
        from raglan.context_builders._shared import build_context

        effective = max_tokens if max_tokens > 0 else self._max_tokens
        return await build_context(candidates, effective, self._loader)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {"max_tokens": self._max_tokens},
        }
