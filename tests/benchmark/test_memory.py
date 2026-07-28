"""Memory stability tests — detect leaks under repeated operations."""

from __future__ import annotations

import asyncio
import gc

import pytest

pytestmark = [pytest.mark.slow]


def _get_rss_mb() -> float:
    """Return current process RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0


# ============================================================================
# BM25 — repeated index/query cycles
# ============================================================================


class TestBM25Memory:
    def test_repeated_index_no_leak(self):
        """10 cycles of build-index → query → rebuild should not grow RSS."""
        from raglan.retrievers.bm25 import BM25Retriever

        gc.collect()
        rss_before = _get_rss_mb()

        for cycle in range(15):
            bm = BM25Retriever()

            async def _cycle(_bm, _cycle_id):
                async def gen():
                    yield [(f"d{i}", f"cycle {_cycle_id} document {i}", None) for i in range(500)]

                await _bm.index(gen())
                await _bm.retrieve(["cycle document"], [], top_k=10)

            asyncio.run(_cycle(bm, cycle))
            del bm

        gc.collect()
        rss_after = _get_rss_mb()

        if rss_before > 0 and rss_after > 0:
            growth = rss_after - rss_before
            # Allow up to 50 MB growth for Python overhead (gc, caches, etc.)
            assert growth < 50, f"RSS grew by {growth:.1f} MB ({rss_before:.0f} → {rss_after:.0f})"


# ============================================================================
# Memory retriever — large batch add/remove
# ============================================================================


class TestMemoryRetrieverMemory:
    def test_add_remove_cycles_stable(self):
        """Repeated add/remove cycles should not accumulate memory."""
        from raglan.retrievers.memory import MemoryRetriever

        gc.collect()
        rss_before = _get_rss_mb()

        mr = MemoryRetriever()

        async def _cycle():
            # Add 200 chunks
            for i in range(200):
                await mr.add([(f"c{i}", f"chunk {i}", None)])
            # Remove half
            await mr.remove([f"c{i}" for i in range(0, 200, 2)])

        for _ in range(20):
            asyncio.run(_cycle())

        gc.collect()
        rss_after = _get_rss_mb()

        if rss_before > 0 and rss_after > 0:
            growth = rss_after - rss_before
            assert growth < 30, f"RSS grew by {growth:.1f} MB ({rss_before:.0f} → {rss_after:.0f})"


# ============================================================================
# Pipeline — repeated search
# ============================================================================


class TestPipelineMemory:
    def test_repeated_search_no_leak(self):
        """500 pipeline searches should not cause unbounded memory growth."""
        from raglan.raglan import Raglan
        from raglan.retrievers.bm25 import BM25Retriever

        bm = BM25Retriever()

        async def _setup():
            async def gen():
                yield [(f"d{i}", f"document {i}", None) for i in range(100)]

            await bm.index(gen())

        asyncio.run(_setup())

        rag = Raglan.builder().with_retrievers([bm]).build()

        gc.collect()
        rss_before = _get_rss_mb()

        async def _search_loop():
            for i in range(500):
                await rag.search(f"document {i % 100}")

        asyncio.run(_search_loop())

        gc.collect()
        rss_after = _get_rss_mb()

        if rss_before > 0 and rss_after > 0:
            growth = rss_after - rss_before
            assert growth < 30, f"RSS grew by {growth:.1f} MB ({rss_before:.0f} → {rss_after:.0f})"


# ============================================================================
# Object count — verify cleanup
# ============================================================================


class TestObjectCleanup:
    def test_bm25_cleanup_after_delete(self):
        """Deleting a BM25Retriever should release its index memory."""
        from raglan.retrievers.bm25 import BM25Retriever

        bm = BM25Retriever()

        async def _build(_bm):
            async def gen():
                yield [(f"d{i}", f"large document {i} " * 50, None) for i in range(1000)]

            await _bm.index(gen())

        asyncio.run(_build(bm))

        # Force collection and count references
        import weakref

        ref = weakref.ref(bm)
        assert ref() is not None, "BM25Retriever still alive before deletion"

        del bm
        gc.collect()
        # BM25Retriever should be garbage-collected after delete
        assert ref() is None, "BM25Retriever not garbage-collected after deletion"
