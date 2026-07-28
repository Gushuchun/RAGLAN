"""Tests for RRF and other fusion strategies."""

from __future__ import annotations

import pytest

from raglan.fusion.round_robin import RoundRobinFusion
from raglan.fusion.rrf import RRFFusion
from raglan.fusion.weighted import WeightedFusion
from raglan.types import ScoredChunk


def _chunk(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, content=f"content-{cid}", score=score)


@pytest.mark.asyncio
async def test_rrf_single_retriever():
    """RRF with one retriever preserves the original ranking order."""
    fusion = RRFFusion()
    results = {
        "bm25": [
            [_chunk("a", 0.9), _chunk("b", 0.7), _chunk("c", 0.3)],
        ]
    }
    fused = await fusion.fuse(results)
    assert len(fused) == 3
    assert fused[0].chunk_id == "a"
    assert fused[1].chunk_id == "b"
    assert fused[2].chunk_id == "c"


@pytest.mark.asyncio
async def test_rrf_deduplicates_by_parent():
    """Chunks sharing a parent are deduplicated — only the best survives."""
    fusion = RRFFusion()
    chunk_a = ScoredChunk(chunk_id="a1", content="a", score=0.9, parent_chunk_id="parent_a")
    chunk_a2 = ScoredChunk(chunk_id="a2", content="a2", score=0.5, parent_chunk_id="parent_a")
    results = {"bm25": [[chunk_a, chunk_a2]]}
    fused = await fusion.fuse(results)
    parent_ids = {c.parent_chunk_id for c in fused}
    assert "parent_a" in parent_ids
    # Only one chunk from parent_a survives
    count = sum(1 for c in fused if c.parent_chunk_id == "parent_a")
    assert count == 1


@pytest.mark.asyncio
async def test_weighted_fusion():
    """Weighted fusion with equal weights averages scores."""
    fusion = WeightedFusion()
    results = {
        "a": [[_chunk("x", 1.0), _chunk("y", 0.0)]],
        "b": [[_chunk("x", 0.0), _chunk("y", 1.0)]],
    }
    fused = await fusion.fuse(results)
    assert len(fused) == 2


@pytest.mark.asyncio
async def test_round_robin_interleaving():
    """Round-robin interleaves results from two retrievers."""
    fusion = RoundRobinFusion()
    results = {
        "a": [[_chunk("a1", 1.0), _chunk("a2", 0.9)]],
        "b": [[_chunk("b1", 0.8), _chunk("b2", 0.7)]],
    }
    fused = await fusion.fuse(results)
    # First chunk from each retriever, then second
    assert fused[0].chunk_id in ("a1", "b1")
    assert fused[1].chunk_id in ("a1", "b1")


@pytest.mark.asyncio
async def test_rrf_dense_weight_path():
    """Non-sparse retriever results use the dense-weight code path."""
    fusion = RRFFusion(dense_weight=1.0, sparse_weight=0.0)
    results = {"pgvector": [[_chunk("a", 0.9), _chunk("b", 0.5)]]}
    fused = await fusion.fuse(results)
    assert len(fused) == 2
    assert fused[0].chunk_id == "a"


@pytest.mark.asyncio
async def test_weighted_fusion_empty_retriever():
    """Weighted fusion skips retrievers that returned no chunks."""
    fusion = WeightedFusion()
    results = {
        "a": [[_chunk("x", 1.0)]],
        "b": [[]],  # empty results
    }
    fused = await fusion.fuse(results)
    assert len(fused) == 1


@pytest.mark.asyncio
async def test_rrf_empty():
    """RRF with no results returns an empty list."""
    fusion = RRFFusion()
    fused = await fusion.fuse({})
    assert fused == []
