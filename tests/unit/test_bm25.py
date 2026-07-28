"""Tests for BM25Retriever."""

from __future__ import annotations

import pytest

from raglan.retrievers.bm25 import BM25Retriever


async def _index_docs(bm: BM25Retriever) -> None:
    async def gen():
        yield [
            ("d1", "how to return a damaged order", None),
            ("d2", "refund policy for online purchases explained", None),
            ("d3", "shipping and delivery timeline estimates", None),
            ("d4", "return policy shipping refund", None),
        ]

    await bm.index(gen())


@pytest.mark.asyncio
async def test_basic_search():
    """BM25 returns the most relevant document first."""
    bm = BM25Retriever()
    await _index_docs(bm)
    results = await bm.retrieve(["return damaged item"], [], top_k=3)
    assert len(results[0]) > 0
    assert results[0][0].chunk_id == "d1"


@pytest.mark.asyncio
async def test_multiple_queries():
    """Batch retrieve returns one result list per query."""
    bm = BM25Retriever()
    await _index_docs(bm)
    results = await bm.retrieve(["return", "shipping"], [], top_k=2)
    assert len(results) == 2
    assert all(len(r) > 0 for r in results)


@pytest.mark.asyncio
async def test_incremental_add():
    """Adding a chunk after indexing makes it searchable."""
    bm = BM25Retriever()
    await _index_docs(bm)
    await bm.add([("d5", "warranty extension for electronics", None)])
    results = await bm.retrieve(["warranty extension"], [], top_k=2)
    assert results[0][0].chunk_id == "d5"


@pytest.mark.asyncio
async def test_incremental_remove():
    """Removing a chunk removes it from search results."""
    bm = BM25Retriever()
    await _index_docs(bm)
    await bm.remove(["d1"])
    results = await bm.retrieve(["return damaged item"], [], top_k=3)
    ids = {c.chunk_id for c in results[0]}
    assert "d1" not in ids


@pytest.mark.asyncio
async def test_remove_nonexistent_chunk():
    """Removing a chunk that does not exist is a no-op, not an error."""
    bm = BM25Retriever()
    await _index_docs(bm)
    await bm.remove(["nonexistent_id"])
    results = await bm.retrieve(["return"], [], top_k=5)
    assert len(results[0]) == 2  # d4 + d1 (d2/d3 don't match "return")


@pytest.mark.asyncio
async def test_chinese_tokenization():
    """Chinese text is tokenized correctly (bigram + whole word)."""
    bm = BM25Retriever()

    async def gen():
        yield [("zh1", "如何退换货 流程说明", None)]

    await bm.index(gen())
    results = await bm.retrieve(["退换货"], [], top_k=3)
    assert len(results[0]) == 1
    assert results[0][0].chunk_id == "zh1"


@pytest.mark.asyncio
async def test_empty_corpus():
    """Searching an empty index returns empty results, not an error."""
    bm = BM25Retriever()
    results = await bm.retrieve(["anything"], [], top_k=5)
    assert results[0] == []


@pytest.mark.asyncio
async def test_idf_zero_for_missing_term():
    """Terms not in the corpus get IDF=0 and contribute no score."""
    bm = BM25Retriever()
    await _index_docs(bm)
    results = await bm.retrieve(["zzz_missing_term_xyz"], [], top_k=3)
    assert results[0] == []
