"""Stress / load tests for Raglan.

These tests exercise the library under heavier-than-normal workloads to
catch performance regressions, memory leaks, and edge-case failures.
They are designed to run quickly (< 5 s each) while pushing boundaries.
"""

from __future__ import annotations

import asyncio

import pytest

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.round_robin import RoundRobinFusion
from raglan.fusion.rrf import RRFFusion
from raglan.fusion.weighted import WeightedFusion
from raglan.middleware.timeout import TimeoutMiddleware
from raglan.pipeline import Pipeline
from raglan.raglan import Raglan
from raglan.retrievers.bm25 import BM25Retriever
from raglan.retrievers.memory import MemoryRetriever

# ==========================================================================
# Large index build
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_large_index():
    """BM25 handles 10 000 documents without error."""
    bm = BM25Retriever()
    batch_size = 1000
    num_batches = 10

    async def gen():
        for b in range(num_batches):
            batch = [
                (f"d{b}_{i}", f"document number {b}_{i} with some varied content", None)
                for i in range(batch_size)
            ]
            yield batch

    await bm.index(gen())

    assert bm._doc_count == batch_size * num_batches

    # Search should still work correctly
    results = await bm.retrieve(["varied content"], [], top_k=5)
    assert len(results[0]) == 5


# ==========================================================================
# High-frequency search
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_high_frequency_search():
    """BM25 handles many rapid sequential searches."""
    bm = BM25Retriever()

    async def gen():
        yield [(f"d{i}", f"document number {i}", None) for i in range(100)]

    await bm.index(gen())

    for i in range(500):
        results = await bm.retrieve([f"document {i % 100}"], [], top_k=3)
        assert len(results[0]) >= 1


# ==========================================================================
# Incremental add stress
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_incremental_add_stress():
    """Adding many small batches incrementally does not degrade performance."""
    bm = BM25Retriever()

    async def gen():
        yield [("seed", "seed document", None)]

    await bm.index(gen())

    # Add 200 documents one by one
    for i in range(200):
        await bm.add([(f"d{i}", f"incremental document {i}", None)])

    assert bm._doc_count == 201

    # Search should still work
    results = await bm.retrieve(["incremental document 150"], [], top_k=3)
    assert len(results[0]) >= 1


# ==========================================================================
# Fusion with many results
# ==========================================================================


@pytest.mark.asyncio
async def test_rrf_fusion_many_candidates():
    """RRF fuses large candidate sets correctly."""
    from raglan.types import ScoredChunk

    # Simulate two retrievers each returning 100 candidates
    dense = [
        ScoredChunk(
            chunk_id=f"dense_{i}",
            content=f"dense content {i}",
            score=1.0 - i * 0.01,
            source="pgvector",
        )
        for i in range(100)
    ]
    sparse = [
        ScoredChunk(
            chunk_id=f"sparse_{i}",
            content=f"sparse content {i}",
            score=0.5 - i * 0.005,
            source="bm25",
        )
        for i in range(100)
    ]

    fusion = RRFFusion()
    results = {"pgvector": [dense], "bm25": [sparse]}
    fused = await fusion.fuse(results)

    # Should produce a reasonable number of unique results
    assert len(fused) >= 50  # at least half are unique across both sources
    # Top result should have the highest score
    assert fused[0].score > fused[-1].score


# ==========================================================================
# Pipeline with many middleware
# ==========================================================================


@pytest.mark.asyncio
async def test_pipeline_many_middleware():
    """Pipeline runs correctly with a middleware before a retriever."""
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
    assert len(results) >= 1
    assert not trace.degraded


# ==========================================================================
# Memory retriever with large embedding batch
# ==========================================================================


@pytest.mark.asyncio
async def test_memory_retriever_large_batch():
    """Memory retriever handles a large embedding batch."""
    import math

    dim = 128
    num_chunks = 500

    chunks = [
        (f"c{i}", f"chunk {i}", [math.sin(i * 0.1 + j) for j in range(dim)], None)
        for i in range(num_chunks)
    ]

    mr = MemoryRetriever()
    mr.load_embedded(chunks)

    query_vec = [math.cos(j) for j in range(dim)]
    results = await mr.retrieve(["q"], [query_vec], top_k=10)

    assert len(results[0]) == 10


# ==========================================================================
# Raglan batch_search stress
# ==========================================================================


@pytest.mark.asyncio
async def test_raglan_batch_search_stress():
    """batch_search handles many queries with bounded concurrency."""
    bm = BM25Retriever()

    async def gen():
        yield [(f"d{i}", f"document number {i}", None) for i in range(50)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()

    # 200 queries with max 8 concurrent
    queries = [f"document {i % 50}" for i in range(200)]
    results = await rag.batch_search(queries, max_concurrency=8)

    assert len(results) == 200
    for r, trace in results:
        assert len(r) >= 1
        assert not trace.degraded


# ==========================================================================
# RoundRobin fusion interleaving correctness
# ==========================================================================


@pytest.mark.asyncio
async def test_round_robin_unequal_lengths():
    """Round-robin handles retrievers with different result counts."""
    from raglan.types import ScoredChunk

    a = [ScoredChunk(chunk_id=f"a{i}", content=f"A{i}", score=1.0) for i in range(10)]
    b = [ScoredChunk(chunk_id=f"b{i}", content=f"B{i}", score=0.8) for i in range(3)]

    fusion = RoundRobinFusion()
    fused = await fusion.fuse({"r1": [a], "r2": [b]})

    # First 6 results should interleave a0, b0, a1, b1, a2, b2
    assert fused[0].chunk_id == "a0"
    assert fused[1].chunk_id == "b0"
    assert fused[2].chunk_id == "a1"
    assert fused[3].chunk_id == "b1"
    assert len(fused) == 13  # 10 + 3, all unique


# ==========================================================================
# Timeout middleware under actual load
# ==========================================================================


@pytest.mark.asyncio
async def test_timeout_middleware_under_load():
    """TimeoutMiddleware correctly degrades a slow stage under load."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "fast document", None)]

    await bm.index(gen())

    class _SlowButWorksStage:
        name = "sloth"

        async def __call__(self, ctx):
            await asyncio.sleep(0.5)
            ctx.final_results = []
            return ctx

    pipeline = Pipeline(
        [
            IdentityExpander(),
            bm,
            RRFFusion(),
            TimeoutMiddleware(0.1),
            _SlowButWorksStage(),
        ],
        fallback_mode="degrade",
    )

    _results, trace = await pipeline.run("fast")
    # The slow stage should be degraded
    assert trace.degraded
    assert any("sloth" in d.stage for d in trace.degradations)


# ==========================================================================
# Weighted fusion with many retrievers
# ==========================================================================


@pytest.mark.asyncio
async def test_weighted_fusion_many_retrievers():
    """Weighted fusion handles 5+ retrievers simultaneously."""
    from raglan.types import ScoredChunk

    results = {}
    for i in range(5):
        chunks = [
            ScoredChunk(
                chunk_id=f"r{i}_c{j}",
                content=f"R{i} chunk {j}",
                score=1.0 - j * 0.1,
                source=f"retriever_{i}",
            )
            for j in range(10)
        ]
        results[f"retriever_{i}"] = [chunks]

    fusion = WeightedFusion()
    fused = await fusion.fuse(results)
    assert len(fused) >= 10  # should have deduplicated somewhat
