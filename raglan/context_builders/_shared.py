"""Shared utilities for context builders — token-budget packing logic."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from raglan.token_utils import count_tokens
from raglan.types import ScoredChunk, SearchResult


async def build_context(
    candidates: list[ScoredChunk],
    max_tokens: int,
    loader: Callable[[list[str]], Awaitable[dict[str, str]]],
    *,
    parent_content_fn: Callable[[ScoredChunk, dict[str, str]], str | None] = lambda c, pm: pm.get(
        c.parent_chunk_id or c.chunk_id
    ),
) -> list[SearchResult]:
    """Build search results, packing parent content into a token budget.

    Parameters
    ----------
    candidates:
        Ranked chunks from the pipeline (fused/reranked).
    max_tokens:
        Hard cap on total tokens in the expanded context.
    loader:
        Async callable ``(chunk_ids: list[str]) -> dict[str, str]`` that
        maps parent IDs to full parent content.
    parent_content_fn:
        Callable ``(candidate, parent_map) -> str | None`` that computes
        the parent content for each candidate.  Default: direct parent
        lookup by ``parent_chunk_id`` or ``chunk_id``.

    Returns
    -------
    list[SearchResult]
        Packed results with parent content, respecting the token budget.
    """
    if not candidates:
        return []

    # Collect unique parent IDs and batch-load content
    parent_ids: set[str] = {c.parent_chunk_id or c.chunk_id for c in candidates}
    parent_map = await loader(list(parent_ids)) if parent_ids else {}

    results: list[SearchResult] = []
    tokens_used = 0

    for c in candidates:
        parent_content = parent_content_fn(c, parent_map)

        if parent_content:
            est = count_tokens(parent_content)
            if tokens_used + est > max_tokens:
                remaining = max_tokens - tokens_used
                if remaining > 0:
                    parent_content = parent_content[: remaining * 4] + "..."
                    tokens_used = max_tokens
            else:
                tokens_used += est

        results.append(
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=c.score,
                parent_content=parent_content,
                metadata=c.chunk_metadata,
                source=c.source,
            )
        )

        if tokens_used >= max_tokens:
            break

    return results
