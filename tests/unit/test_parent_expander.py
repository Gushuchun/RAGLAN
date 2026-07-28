"""Tests for ParentExpander."""

from __future__ import annotations

import pytest

from raglan.context_builders.parent_expander import ParentExpander
from raglan.types import ScoredChunk


async def _loader(chunk_ids: list[str]) -> dict[str, str]:
    return {
        "p_a": "A" * 120,  # ~40 tokens
        "p_b": "B" * 300,  # ~100 tokens
        "p_c": "C" * 6000,  # ~2000 tokens — will exceed limit
    }


@pytest.mark.asyncio
async def test_parent_expander_loads_parents():
    expander = ParentExpander(_loader, max_tokens=9999)
    candidates = [
        ScoredChunk(chunk_id="c1", content="child", score=0.9, parent_chunk_id="p_a"),
    ]
    results = await expander.build("query", candidates)
    assert results[0].parent_content == "A" * 120


@pytest.mark.asyncio
async def test_parent_expander_empty_candidates():
    expander = ParentExpander(_loader)
    results = await expander.build("query", [])
    assert results == []


@pytest.mark.asyncio
async def test_parent_expander_no_parent_chunk_id():
    """Chunks without parent_chunk_id use their own chunk_id to look up parent."""
    expander = ParentExpander(_loader, max_tokens=9999)
    candidates = [
        ScoredChunk(chunk_id="p_a", content="self", score=0.9, parent_chunk_id=None),
    ]
    results = await expander.build("query", candidates)
    assert results[0].parent_content == "A" * 120  # chunk_id used as parent_id


@pytest.mark.asyncio
async def test_parent_expander_truncates_at_token_limit():
    """When token limit is exceeded, the parent content is truncated."""
    expander = ParentExpander(_loader, max_tokens=2)  # 2 tokens
    candidates = [
        ScoredChunk(chunk_id="c1", content="child", score=0.9, parent_chunk_id="p_a"),
    ]
    results = await expander.build("query", candidates, max_tokens=-1)
    assert results[0].parent_content is not None
    # "A" * 100 = 13 tiktoken tokens, > 2 → truncate
    # remaining=2, truncation: 2*4 chars + "..." = 11 chars
    assert len(results[0].parent_content) == 11  # "AAAAAAAA..."


@pytest.mark.asyncio
async def test_parent_expander_greedy_fill():
    """First few results get parent content; later ones are truncated."""
    expander = ParentExpander(_loader, max_tokens=50)  # ~150 chars
    candidates = [
        ScoredChunk(chunk_id="c1", content="1", score=0.9, parent_chunk_id="p_a"),  # 120 chars
        ScoredChunk(chunk_id="c2", content="2", score=0.8, parent_chunk_id="p_b"),  # 300 chars
    ]
    results = await expander.build("query", candidates)
    # First result gets full parent (120 chars < 150 limit)
    assert results[0].parent_content == "A" * 120
    # Second result gets truncated (only ~30 chars left)
    assert results[1].parent_content is not None
    assert len(results[1].parent_content) < 300


@pytest.mark.asyncio
async def test_parent_expander_uses_runtime_max_tokens():
    """Runtime max_tokens overrides constructor default."""
    expander = ParentExpander(_loader, max_tokens=9999)
    candidates = [
        ScoredChunk(chunk_id="c1", content="child", score=0.9, parent_chunk_id="p_c"),
    ]
    results = await expander.build("query", candidates, max_tokens=5)
    # p_c is 6000 chars but with max_tokens=5 we get ~15 chars + "..."
    assert len(results[0].parent_content) < 6000
