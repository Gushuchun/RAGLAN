"""Tests for the Pipeline engine."""

from __future__ import annotations

import pytest

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.pipeline import Pipeline
from raglan.retrievers.bm25 import BM25Retriever


async def _indexed_pipeline():
    bm = BM25Retriever()
    batch = [
        [
            ("d1", "how to return a damaged order", None),
            ("d2", "shipping policy and delivery times", None),
        ]
    ]

    async def gen():
        for b in batch:
            yield b

    await bm.index(gen())
    return Pipeline(
        [
            IdentityExpander(),
            bm,
            RRFFusion(),
            PassthroughBuilder(),
        ]
    )


@pytest.mark.asyncio
async def test_basic_pipeline_run():
    """A complete pipeline returns results and a trace."""
    pipeline = await _indexed_pipeline()
    results, trace = await pipeline.run("return order")
    assert len(results) >= 1
    assert trace.query == "return order"
    assert trace.total_ms >= 0
    assert not trace.degraded


@pytest.mark.asyncio
async def test_pipeline_stage_timing():
    """Every stage produces a timing record."""
    pipeline = await _indexed_pipeline()
    _results, trace = await pipeline.run("return order")
    assert len(trace.stage_timings) == 4  # identity, bm25, rrf, passthrough
    stage_names = [t.stage for t in trace.stage_timings]
    assert "identity" in stage_names
    assert "bm25" in stage_names
    assert "rrf" in stage_names
    assert "passthrough" in stage_names


@pytest.mark.asyncio
async def test_pipeline_degradation_strict_mode():
    """In strict mode a stage failure propagates."""

    class _FailingExpander:
        name = "failing"

        async def expand(self, query, num_variants=3):
            raise RuntimeError("LLM is down")

    pipeline = Pipeline(
        [_FailingExpander(), PassthroughBuilder()],
        fallback_mode="strict",
    )
    with pytest.raises(RuntimeError, match="LLM is down"):
        await pipeline.run("test query")


@pytest.mark.asyncio
async def test_pipeline_degradation_degrade_mode():
    """In degrade mode a stage failure records a degradation and continues."""

    class _FailingExpander:
        name = "failing"

        async def expand(self, query, num_variants=3):
            raise RuntimeError("LLM is down")

    pipeline = Pipeline(
        [_FailingExpander(), PassthroughBuilder()],
        fallback_mode="degrade",
    )
    _results, trace = await pipeline.run("test query")
    assert trace.degraded
    assert "failing" in trace.degraded_stage_names


@pytest.mark.asyncio
async def test_pipeline_empty_query():
    """Empty query should not crash."""
    pipeline = await _indexed_pipeline()
    results, trace = await pipeline.run("")
    # Empty query to BM25 returns nothing, which is fine
    assert isinstance(results, list)
    assert not trace.degraded
