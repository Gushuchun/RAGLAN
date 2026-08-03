"""Performance benchmarks for critical paths.

Uses pytest-benchmark to track performance regressions.
Run with: pytest tests/test_benchmark.py --benchmark-only
"""

from __future__ import annotations

import asyncio
import math

import pytest

pytestmark = [pytest.mark.slow]


# ============================================================================
# BM25 benchmarks
# ============================================================================


class TestBM25Benchmark:
    @pytest.fixture(scope="class")
    def bm25_index(self):
        """Build a 10k-document BM25 index (shared across benchmarks)."""
        from raglan.retrievers.bm25 import BM25Retriever

        bm = BM25Retriever()

        async def _build():
            async def gen():
                for b in range(10):
                    yield [
                        (f"d{b}_{i}", f"document number {b}_{i} with varied content words", None)
                        for i in range(1000)
                    ]

            await bm.index(gen())

        asyncio.run(_build())
        return bm

    def test_index_10k_docs(self, benchmark):
        """Benchmark: build a 10 000-document index."""
        from raglan.retrievers.bm25 import BM25Retriever

        def _do():
            bm = BM25Retriever()

            async def _index():
                async def gen():
                    yield [
                        (f"d{i}", f"benchmark document {i} with some content", None)
                        for i in range(1000)
                    ]

                await bm.index(gen())

            asyncio.run(_index())
            return bm._index._doc_count

        result = benchmark(_do)
        assert result == 1000

    def test_search_throughput(self, benchmark, bm25_index):
        """Benchmark: single BM25 query latency."""
        bm = bm25_index

        def _search():
            return asyncio.run(bm.retrieve(["varied content words"], [], top_k=10))

        result = benchmark(_search)
        assert len(result[0]) == 10

    def test_batch_search_throughput(self, benchmark, bm25_index):
        """Benchmark: 100 sequential searches."""
        bm = bm25_index

        def _search_batch():
            async def _go():
                for i in range(100):
                    await bm.retrieve([f"document {i % 1000}"], [], top_k=5)

            asyncio.run(_go())

        benchmark(_search_batch)


# ============================================================================
# Fusion benchmarks
# ============================================================================


class TestFusionBenchmark:
    def test_rrf_fusion_1k_candidates(self, benchmark):
        """Benchmark: RRF fusion of 2 x 500 candidates."""
        from raglan.fusion.rrf import RRFFusion
        from raglan.types import ScoredChunk

        dense = [ScoredChunk(f"d{i}", f"dense {i}", 1.0 - i * 0.001) for i in range(500)]
        sparse = [ScoredChunk(f"s{i}", f"sparse {i}", 0.5 - i * 0.0005) for i in range(500)]

        fusion = RRFFusion()

        def _fuse():
            return asyncio.run(fusion.fuse({"pgvector": [dense], "bm25": [sparse]}))

        result = benchmark(_fuse)
        assert len(result) >= 100


# ============================================================================
# Memory retriever benchmarks
# ============================================================================


class TestMemoryRetrieverBenchmark:
    @pytest.fixture(scope="class")
    def mem_index(self):
        from raglan.retrievers.memory import MemoryRetriever

        dim = 128
        mr = MemoryRetriever()
        mr.load_embedded(
            [
                (f"c{i}", f"chunk {i}", [math.sin(i * 0.1 + j) for j in range(dim)], None)
                for i in range(1000)
            ]
        )
        return mr

    def test_cosine_search_1k_128d(self, benchmark, mem_index):
        """Benchmark: cosine search over 1000 x 128-dim vectors."""
        mr = mem_index
        query_vec = [math.cos(j) for j in range(128)]

        def _search():
            return asyncio.run(mr.retrieve(["q"], [query_vec], top_k=10))

        result = benchmark(_search)
        assert len(result[0]) == 10


# ============================================================================
# Pipeline benchmarks
# ============================================================================


class TestPipelineBenchmark:
    def test_minimal_pipeline_end_to_end(self, benchmark):
        """Benchmark: minimal pipeline (BM25 only) end-to-end latency."""
        from raglan.raglan import Raglan
        from raglan.retrievers.bm25 import BM25Retriever

        bm = BM25Retriever()

        async def _setup():
            async def gen():
                yield [(f"d{i}", f"document {i}", None) for i in range(1000)]

            await bm.index(gen())

        asyncio.run(_setup())
        rag = Raglan.builder().with_retrievers([bm]).build()

        def _search():
            return asyncio.run(rag.search("document 42"))

        result = benchmark(_search)
        assert len(result[0]) >= 1


# ============================================================================
# Tokenizer benchmark
# ============================================================================


class TestTokenizerBenchmark:
    def test_builtin_tokenizer_throughput(self, benchmark):
        """Benchmark: tokeniser on 10 KB of mixed English + CJK text."""
        from raglan.retrievers.bm25 import BM25Retriever

        # Build ~10 KB of mixed text
        en = "The quick brown fox jumps over the lazy dog. " * 50
        zh = "人工智能正在改变世界。检索增强生成系统是未来。" * 20
        text = en + zh

        tok = BM25Retriever._builtin_tokenizer

        def _tokenize():
            return tok(text)

        result = benchmark(_tokenize)
        assert len(result) > 100
