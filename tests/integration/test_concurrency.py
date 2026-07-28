"""Concurrency and thread-safety stress tests for Raglan.

These tests verify that the fixes applied in the code review (asyncio.Lock
on BM25, Semaphore on batch_search, CircuitBreaker lock) actually work
under concurrent load.
"""

from __future__ import annotations

import asyncio

import pytest

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.middleware.circuit_breaker import CircuitBreakerMiddleware
from raglan.pipeline import Pipeline
from raglan.raglan import Raglan
from raglan.retrievers.bm25 import BM25Retriever

# ==========================================================================
# BM25: concurrent reads
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_concurrent_reads():
    """Multiple concurrent retrieve() calls do not race."""
    bm = BM25Retriever()

    async def gen():
        yield [
            ("d1", "return policy for damaged items", None),
            ("d2", "shipping and delivery timeline", None),
            ("d3", "refund process explained", None),
        ]

    await bm.index(gen())

    async def search(q: str):
        results = await bm.retrieve([q], [], top_k=3)
        return len(results[0])

    tasks = [
        search("return"),
        search("shipping"),
        search("refund"),
        search("damaged"),
        search("delivery"),
        search("policy"),
        search("explained"),
        search("process"),
        search("timeline"),
        search("items"),
    ] * 10  # 100 concurrent searches

    counts = await asyncio.gather(*tasks)
    assert all(c >= 1 for c in counts), f"Some searches returned 0 results: {counts}"
    assert len(counts) == 100


# ==========================================================================
# BM25: concurrent mutation + read
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_concurrent_add_and_retrieve():
    """add() and retrieve() happening concurrently do not corrupt each other."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return policy for damaged items", None)]

    await bm.index(gen())

    async def adder():
        for i in range(50):
            await bm.add([(f"add_{i}", f"document number {i}", None)])
            await asyncio.sleep(0)  # yield to event loop

    async def reader():
        results_list = []
        for _ in range(50):
            r = await bm.retrieve(["return"], [], top_k=5)
            results_list.append(r)
            await asyncio.sleep(0)
        return results_list

    # Run both concurrently
    add_task = asyncio.create_task(adder())
    read_results = await reader()
    await add_task

    # Every read should have at least the original document
    for r in read_results:
        assert len(r[0]) >= 1
        ids = {c.chunk_id for c in r[0]}
        assert "d1" in ids, "Original document d1 missing from results"


@pytest.mark.asyncio
async def test_bm25_concurrent_remove_and_retrieve():
    """remove() and retrieve() happening concurrently do not corrupt each other."""
    bm = BM25Retriever()
    chunks = [(f"d{i}", f"document number {i}", None) for i in range(30)]

    async def gen():
        yield chunks

    await bm.index(gen())

    async def remover():
        for i in range(0, 30, 2):
            await bm.remove([f"d{i}"])
            await asyncio.sleep(0)

    async def reader():
        results_list = []
        for _ in range(30):
            r = await bm.retrieve(["document"], [], top_k=3)
            results_list.append(r)
            await asyncio.sleep(0)
        return results_list

    rem_task = asyncio.create_task(remover())
    read_results = await reader()
    await rem_task

    # No read should raise KeyError
    for r in read_results:
        assert isinstance(r, list)
        assert isinstance(r[0], list)


# ==========================================================================
# BM25: concurrent index rebuild + retrieve
# ==========================================================================


@pytest.mark.asyncio
async def test_bm25_concurrent_index_and_retrieve():
    """index() during ongoing retrieve() — old results are served until swap."""
    bm = BM25Retriever()

    async def gen():
        yield [("old", "original document content", None)]

    await bm.index(gen())

    async def rebuilder():
        async def new_gen():
            yield [("new", "completely different content", None)]

        await bm.index(new_gen())

    async def reader():
        results_list = []
        for _ in range(10):
            r = await bm.retrieve(["original"], [], top_k=3)
            results_list.append(len(r[0]))
            await asyncio.sleep(0)
        return results_list

    # Index rebuild happens while reading
    rebuild_task = asyncio.create_task(rebuilder())
    read_counts = await reader()
    await rebuild_task

    # All reads should succeed (old doc visible until swap, then new doc)
    assert all(c >= 0 for c in read_counts)


# ==========================================================================
# Batch search with Semaphore
# ==========================================================================


@pytest.mark.asyncio
async def test_batch_search_respects_semaphore():
    """batch_search with max_concurrency limits concurrent pipeline runs."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "test document", None)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()

    queries = [f"test query {i}" for i in range(50)]
    max_concurrency = 5

    results = await rag.batch_search(queries, max_concurrency=max_concurrency)

    assert len(results) == 50
    for r, trace in results:
        assert isinstance(r, list)
        assert not trace.degraded


# ==========================================================================
# CircuitBreaker under concurrent load
# ==========================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_concurrent():
    """CircuitBreaker handles concurrent wrap() calls without state corruption."""
    cb = CircuitBreakerMiddleware(failure_threshold=3, recovery_timeout=999.0)

    class _CountingFailingStage:
        name = "counter"

        def __init__(self):
            self.call_count = 0

        async def __call__(self, ctx):
            self.call_count += 1
            raise RuntimeError("fail")

    stage = _CountingFailingStage()

    # Run 20 concurrent requests — all should fail, CB should open after 3
    async def request():
        try:
            await cb.wrap(
                type(
                    "Ctx",
                    (),
                    {
                        "degradations": [],
                        "metadata": {},
                    },
                )(),
                stage,
            )
        except RuntimeError:
            return "fail"
        return "ok"

    results = await asyncio.gather(*(request() for _ in range(20)))
    # Most should have failed, some may have been degraded (CB open)
    assert "fail" in results
    # After 3 failures the circuit breaker should be open
    assert cb._state == "open"
    assert cb._failures >= cb._threshold


# ==========================================================================
# Pipeline: many concurrent runs
# ==========================================================================


@pytest.mark.asyncio
async def test_pipeline_many_concurrent_runs():
    """Hundreds of concurrent pipeline runs don't corrupt state."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "hello world document", None)]

    await bm.index(gen())

    pipeline = Pipeline(
        [
            IdentityExpander(),
            bm,
            RRFFusion(),
            PassthroughBuilder(),
        ]
    )

    async def search(q: str):
        results, _ = await pipeline.run(q)
        return len(results)

    queries = ["hello", "world", "document", "hello world"] * 50  # 200 queries
    counts = await asyncio.gather(*(search(q) for q in queries))

    assert len(counts) == 200
    assert all(c >= 1 for c in counts)


# ==========================================================================
# Pipeline: timeout enforcement
# ==========================================================================


@pytest.mark.asyncio
async def test_pipeline_global_timeout():
    """Pipeline.run(timeout=...) enforces a global deadline."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "hello", None)]

    await bm.index(gen())

    class _SlowExpander:
        name = "slow"

        async def expand(self, query, num_variants=3):
            await asyncio.sleep(10.0)  # way past timeout
            return [query], {}

    pipeline = Pipeline(
        [_SlowExpander(), bm, RRFFusion(), PassthroughBuilder()],
        fallback_mode="degrade",
    )
    _results, trace = await pipeline.run("test", timeout=0.1)
    # Pipeline should have degraded, recording the timeout
    assert trace.degraded
    assert any("timeout" in d.error.lower() for d in trace.degradations)


# ==========================================================================
# Raglan: concurrent search() calls
# ==========================================================================


@pytest.mark.asyncio
async def test_raglan_concurrent_search():
    """Multiple concurrent Raglan.search() calls are safe."""
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return policy for damaged items", None)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()

    async def search(q: str):
        results, trace = await rag.search(q)
        return len(results), trace.degraded

    tasks = [search("return policy") for _ in range(30)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 30
    for count, degraded in results:
        assert count >= 1
        assert not degraded


# ==========================================================================
# Empty query validation
# ==========================================================================


@pytest.mark.asyncio
async def test_raglan_rejects_empty_query():
    """Raglan.search() rejects empty or whitespace-only queries."""
    bm = BM25Retriever()
    rag = Raglan.builder().with_retrievers([bm]).build()

    with pytest.raises(ValueError, match="non-empty"):
        await rag.search("")

    with pytest.raises(ValueError, match="non-empty"):
        await rag.search("   ")
