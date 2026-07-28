"""Tests for Raglan Builder and Facade."""

from __future__ import annotations

import pytest

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.embedders.openai import OpenAIEmbedder  # noqa — imported for presence check
from raglan.exceptions import ConfigurationError
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.raglan import Raglan, RaglanBuilder
from raglan.retrievers.bm25 import BM25Retriever
from raglan.retrievers.memory import MemoryRetriever

# ---------------------------------------------------------------------------
# Builder validation
# ---------------------------------------------------------------------------


def test_builder_rejects_no_retrievers():
    with pytest.raises(ConfigurationError, match="At least one Retriever"):
        Raglan.builder().build()


def test_builder_rejects_dense_retriever_without_embedder():
    with pytest.raises(ConfigurationError, match=r"retriever.*require.*embedding"):
        Raglan.builder().with_retrievers([MemoryRetriever()]).build()


def test_builder_accepts_bm25_without_embedder():
    bm = BM25Retriever()
    rag = Raglan.builder().with_retrievers([bm]).build()
    assert isinstance(rag, Raglan)


def test_builder_applies_defaults():
    bm = BM25Retriever()
    rag = Raglan.builder().with_retrievers([bm]).build()
    assert rag._pipeline._items  # pipeline was assembled


def test_builder_with_all_stages():
    bm = BM25Retriever()
    rag = (
        Raglan.builder()
        .with_expander(IdentityExpander())
        .with_retrievers([bm])
        .with_fusion(RRFFusion())
        .with_context_builder(PassthroughBuilder())
        .with_fallback_mode("degrade")
        .build()
    )
    assert isinstance(rag, Raglan)


def test_builder_rejects_invalid_fallback_mode():
    bm = BM25Retriever()
    with pytest.raises(ConfigurationError, match="fallback_mode"):
        Raglan.builder().with_retrievers([bm]).with_fallback_mode("panic").build()


# ---------------------------------------------------------------------------
# Raglan Facade — end-to-end search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facade_search():
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "how to return a damaged item", None)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()
    results, trace = await rag.search("return item")
    assert len(results) >= 1
    assert trace.query == "return item"
    assert not trace.degraded


@pytest.mark.asyncio
async def test_facade_batch_search():
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return item", None), ("d2", "shipping info", None)]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()
    batch = await rag.batch_search(["return", "shipping"])
    assert len(batch) == 2
    for results, trace in batch:
        assert isinstance(results, list)
        assert isinstance(trace.query, str)


@pytest.mark.asyncio
async def test_facade_search_with_top_k_override():
    bm = BM25Retriever()

    async def gen():
        yield [
            ("d1", "return policy for damaged items", None),
            ("d2", "return shipping label request", None),
            ("d3", "how to return an order", None),
        ]

    await bm.index(gen())

    rag = Raglan.builder().with_retrievers([bm]).build()
    results, _ = await rag.search("return", top_k=1)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Builder method chaining
# ---------------------------------------------------------------------------


def test_builder_methods_return_self():
    b = Raglan.builder()
    assert b.with_expander(IdentityExpander()) is b
    assert b.with_retrievers([BM25Retriever()]) is b
    assert b.with_fusion(RRFFusion()) is b
    assert b.with_context_builder(PassthroughBuilder()) is b
    assert b.with_fallback_mode("strict") is b


# ---------------------------------------------------------------------------
# Raglan static constructors
# ---------------------------------------------------------------------------


def test_builder_static_method():
    assert isinstance(Raglan.builder(), RaglanBuilder)
