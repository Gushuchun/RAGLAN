"""Tests for MemoryRetriever."""

from __future__ import annotations

import pytest

from raglan.retrievers.memory import MemoryRetriever


@pytest.mark.asyncio
async def test_empty_retriever():
    mr = MemoryRetriever()
    results = await mr.retrieve(["test"], [[0.1, 0.2]], top_k=5)
    assert results[0] == []


@pytest.mark.asyncio
async def test_basic_search():
    mr = MemoryRetriever()
    mr.load_embedded(
        [
            ("a", "hello world", [1.0, 0.0, 0.0], None),
            ("b", "goodbye universe", [0.0, 1.0, 0.0], None),
        ]
    )
    results = await mr.retrieve(["ignored"], [[1.0, 0.0, 0.0]], top_k=2)
    assert len(results[0]) == 2
    assert results[0][0].chunk_id == "a"  # closest cosine


@pytest.mark.asyncio
async def test_multiple_queries():
    mr = MemoryRetriever()
    mr.load_embedded(
        [
            ("a", "hello", [1.0, 0.0], None),
            ("b", "world", [0.0, 1.0], None),
        ]
    )
    results = await mr.retrieve(
        ["q1", "q2"],
        [[1.0, 0.0], [0.0, 1.0]],
        top_k=1,
    )
    assert len(results) == 2
    assert results[0][0].chunk_id == "a"
    assert results[1][0].chunk_id == "b"


@pytest.mark.asyncio
async def test_incremental_add():
    mr = MemoryRetriever()
    mr.load_embedded([("a", "hello", [1.0, 0.0], None)])
    await mr.add([("b", "world", {"lang": "en"})])
    # After add, the chunk has no embedding (empty list), so cosine=0
    results = await mr.retrieve(["q"], [[1.0, 0.0]], top_k=2)
    assert len(results[0]) == 2


@pytest.mark.asyncio
async def test_incremental_remove():
    mr = MemoryRetriever()
    mr.load_embedded(
        [
            ("a", "hello", [1.0, 0.0], None),
            ("b", "world", [0.0, 1.0], None),
        ]
    )
    await mr.remove(["a"])
    results = await mr.retrieve(["q"], [[1.0, 0.0]], top_k=2)
    ids = {c.chunk_id for c in results[0]}
    assert "a" not in ids
    assert "b" in ids


@pytest.mark.asyncio
async def test_index_replaces_all():
    mr = MemoryRetriever()
    mr.load_embedded([("a", "old", [1.0, 0.0], None)])

    async def gen():
        yield [("b", "new", {"replaced": True})]

    await mr.index(gen())
    results = await mr.retrieve(["q"], [[1.0, 0.0]], top_k=5)
    ids = {c.chunk_id for c in results[0]}
    assert "a" not in ids
    assert "b" in ids


@pytest.mark.asyncio
async def test_cosine_perfect_match():
    mr = MemoryRetriever()
    mr.load_embedded([("a", "content", [3.0, 4.0], None)])
    results = await mr.retrieve(["q"], [[3.0, 4.0]], top_k=1)
    assert results[0][0].score == pytest.approx(1.0, abs=0.001)


@pytest.mark.asyncio
async def test_cosine_orthogonal():
    mr = MemoryRetriever()
    mr.load_embedded([("a", "x", [1.0, 0.0], None)])
    results = await mr.retrieve(["q"], [[0.0, 1.0]], top_k=1)
    assert results[0][0].score == pytest.approx(0.0, abs=0.001)
