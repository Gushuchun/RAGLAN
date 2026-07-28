"""Integration tests — full end-to-end Raglan pipeline with Builder."""

from __future__ import annotations

import pytest

from raglan import SearchResult
from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.middleware.timeout import TimeoutMiddleware
from raglan.pipeline import Pipeline
from raglan.raglan import Raglan
from raglan.retrievers.bm25 import BM25Retriever

# ---------------------------------------------------------------------------
# Full pipeline with Builder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_builder():
    """End-to-end: BM25 + RRF + Passthrough via Builder."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return policy for damaged items", None)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()
    results, trace = await rag.search("return damaged")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk_id == "d1"
    assert trace.total_ms >= 0
    assert not trace.degraded
    assert len(trace.stage_timings) == 4  # identity, bm25, rrf, passthrough


# ---------------------------------------------------------------------------
# Full pipeline with Programmatic API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_programmatic():
    """End-to-end: Pipeline constructed manually, all stages explicit."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "refund policy explained in detail", None)]

    await bm.index(gen())

    pipeline = Pipeline(
        [
            IdentityExpander(),
            bm,
            RRFFusion(),
            PassthroughBuilder(),
        ]
    )
    results, _trace = await pipeline.run("refund request")
    assert len(results) == 1
    assert results[0].chunk_id == "d1"


# ---------------------------------------------------------------------------
# Degradation end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degradation_flow():
    """A failing expander degrades gracefully; pipeline continues with degradation recorded."""

    class _FailingExpander:
        name = "unstable_expander"

        async def expand(self, query, num_variants=3):
            raise ConnectionError("LLM is unreachable")

    bm = BM25Retriever()

    async def gen():
        yield [("d1", "test document content", None)]

    await bm.index(gen())

    pipeline = Pipeline(
        [_FailingExpander(), bm, RRFFusion(), PassthroughBuilder()],
        fallback_mode="degrade",
    )
    _results, trace = await pipeline.run("test")

    # Degradation is recorded — the expander failed but pipeline continued
    assert trace.degraded
    assert "unstable_expander" in trace.degraded_stage_names


# ---------------------------------------------------------------------------
# Middleware + Pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_in_pipeline():
    """Timeout middleware in a real pipeline does not affect successful runs."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "hello world", None)]

    await bm.index(gen())

    pipeline = Pipeline(
        [
            IdentityExpander(),
            TimeoutMiddleware(5.0),
            bm,
            RRFFusion(),
            PassthroughBuilder(),
        ]
    )
    results, trace = await pipeline.run("hello")
    assert len(results) == 1
    assert not trace.degraded


# ---------------------------------------------------------------------------
# Raglan batch_search integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_search_integration():
    """Batch search runs multiple queries concurrently."""
    bm = BM25Retriever()

    async def gen():
        yield [
            ("d1", "return policy", None),
            ("d2", "shipping info", None),
            ("d3", "refund process", None),
        ]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()
    results = await rag.batch_search(["return", "shipping", "refund"])

    assert len(results) == 3
    for r, t in results:
        assert len(r) >= 1
        assert isinstance(t.query, str)
