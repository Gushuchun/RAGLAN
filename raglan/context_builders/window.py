"""Sliding-window context builder — expands each chunk with surrounding text."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from raglan.types import ScoredChunk, SearchResult

ParentChunkLoader = Callable[[list[str]], Awaitable[dict[str, str]]]


async def _empty_loader(ids: list[str]) -> dict[str, str]:
    """No-op loader used when no user loader is configured."""
    return {}


class WindowBuilder:
    """Wraps each chunk with *window_chars* of surrounding context from its parent.

    Parameters
    ----------
    loader:
        Async callable ``(chunk_ids: list[str]) -> dict[str, str]`` that
        maps child-chunk IDs to full parent-document content.
    window_chars:
        Number of characters of surrounding context to include on each side
        of the matched chunk.  Default 500.
    max_tokens:
        Hard cap on total tokens in the expanded context.  Exceeding
        chunks are truncated.
    """

    name = "window"

    def __init__(
        self,
        loader: ParentChunkLoader | None = None,
        *,
        window_chars: int = 500,
        max_tokens: int = 6000,
    ) -> None:
        self._loader = loader
        self._window = window_chars
        self._max_tokens = max_tokens

    async def build(
        self,
        query: str,
        candidates: list[ScoredChunk],
        max_tokens: int = -1,
    ) -> list[SearchResult]:
        from raglan.context_builders._shared import build_context

        effective = max_tokens if max_tokens > 0 else self._max_tokens

        # Use a no-op loader when none is configured
        loader = self._loader or _empty_loader

        def _window_for(candidate: ScoredChunk, parent_map: dict[str, str]) -> str | None:
            pid = candidate.parent_chunk_id or candidate.chunk_id
            parent_content = parent_map.get(pid)
            if not parent_content:
                return None
            return self._extract_window(candidate.content, parent_content)

        return await build_context(candidates, effective, loader, parent_content_fn=_window_for)

    def _extract_window(self, chunk_content: str, parent_content: str) -> str:
        """Extract a window of text around *chunk_content* within *parent_content*.

        If the chunk is found, returns parent text with *window_chars* of
        surrounding context on each side.  If not found, returns the first
        *window_chars* * 2 characters of the parent.
        """
        idx = parent_content.find(chunk_content)
        if idx == -1:
            # Chunk not found verbatim — return beginning of parent
            return parent_content[: self._window * 2]

        start = max(0, idx - self._window)
        end = min(len(parent_content), idx + len(chunk_content) + self._window)
        return parent_content[start:end]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "window_chars": self._window,
                "max_tokens": self._max_tokens,
            },
        }
