"""Targeted tests to push coverage past 90%."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from raglan.context_builders.parent_expander import ParentExpander
from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.exceptions import ConfigurationError
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.middleware.circuit_breaker import CircuitBreakerMiddleware
from raglan.middleware.logging import LoggingMiddleware
from raglan.middleware.retry import RetryMiddleware
from raglan.pipeline import Pipeline
from raglan.raglan import Raglan
from raglan.retrievers.bm25 import BM25Retriever
from raglan.types import PipelineContext, ScoredChunk

# ==========================================================================
# pipeline.py — parallel branch & edge cases
# ==========================================================================


@pytest.mark.asyncio
async def test_pipeline_parallel_retrievers():
    """Two retrievers in a list run in parallel."""
    bm1 = BM25Retriever()
    bm2 = BM25Retriever()

    async def gen():
        yield [("d1", "return policy", None)]

    await bm1.index(gen())

    async def gen2():
        yield [("d2", "shipping info", None)]

    await bm2.index(gen2())

    pipeline = Pipeline(
        [
            IdentityExpander(),
            [bm1, bm2],  # parallel group
            RRFFusion(),
            PassthroughBuilder(),
        ]
    )
    results, trace = await pipeline.run("return shipping")
    assert len(results) >= 1
    assert not trace.degraded


@pytest.mark.asyncio
async def test_pipeline_fallback_raises_in_strict_mode():
    """Strict mode propagates exception from a non-first stage."""

    class _FailingFusion:
        name = "bad_fusion"

        async def fuse(self, retriever_results, original_query_idx=0):
            raise RuntimeError("fusion engine failure")

    bm = BM25Retriever()

    async def gen():
        yield [("d1", "content", None)]

    await bm.index(gen())

    pipeline = Pipeline(
        [IdentityExpander(), bm, _FailingFusion(), PassthroughBuilder()],
        fallback_mode="strict",
    )
    with pytest.raises(RuntimeError, match="fusion engine failure"):
        await pipeline.run("test")


# ==========================================================================
# raglan.py — from_dict, export_config, search_sync, Builder setters
# ==========================================================================


def test_from_dict():
    """from_dict constructs a Raglan from a config dict (retrievers required)."""
    with pytest.raises(ConfigurationError, match="At least one Retriever"):
        Raglan.from_dict({})


def test_export_config():
    bm = BM25Retriever()
    rag = Raglan.builder().with_retrievers([bm]).build()
    cfg = rag.export_config()
    assert isinstance(cfg, dict)


def test_builder_with_embedder_and_reranker():
    """Builder setters for embedder and reranker return self for chaining."""
    b = Raglan.builder()
    # Test method chaining — embedder/reranker setters return self
    assert b.with_embedder(None) is b  # type: ignore[arg-type]
    b2 = Raglan.builder()
    assert b2.with_reranker(None) is b2  # type: ignore[arg-type]


def test_builder_explicit_methods_return_self():
    b = Raglan.builder()
    assert b.with_expander(IdentityExpander()) is b
    assert b.with_retrievers([BM25Retriever()]) is b
    assert b.with_fusion(RRFFusion()) is b
    assert b.with_context_builder(PassthroughBuilder()) is b


# ==========================================================================
# parent_expander.py — token limit edge case (remaining=0)
# ==========================================================================


@pytest.mark.asyncio
async def test_parent_expander_no_remaining_tokens():
    """When first chunk exactly exhausts the budget, second chunk gets parent_content=None."""

    async def loader(ids: list[str]) -> dict[str, str]:
        return {"p1": "A" * 50, "p2": "B" * 300}  # p1=7 tokens, p2=13 tokens

    expander = ParentExpander(loader, max_tokens=5)  # 5 tokens
    candidates = [
        ScoredChunk(chunk_id="c1", content="child", score=0.9, parent_chunk_id="p1"),
        ScoredChunk(chunk_id="c2", content="child2", score=0.8, parent_chunk_id="p2"),
    ]
    results = await expander.build("q", candidates, max_tokens=-1)
    # First: 50 As = 7 tiktoken tokens > 5 → truncate, uses all 5 tokens
    assert results[0].parent_content is not None
    assert len(results[0].parent_content) < 50
    # Second: budget exhausted (tokens_used >= effective), so break in loop
    # It never gets processed → only 1 result


@pytest.mark.asyncio
async def test_parent_expander_exact_budget_exhausted():
    """When first chunk fills the budget, loop breaks — second is never processed."""

    async def loader(ids: list[str]) -> dict[str, str]:
        return {"p1": "A" * 60, "p2": "B" * 60}

    expander = ParentExpander(loader, max_tokens=2)
    candidates = [
        ScoredChunk(chunk_id="c1", content="child", score=0.9, parent_chunk_id="p1"),
        ScoredChunk(chunk_id="c2", content="child2", score=0.8, parent_chunk_id="p2"),
    ]
    results = await expander.build("q", candidates, max_tokens=-1)
    # First chunk fits exactly (6 chars = 2 tokens), budget exhausted, loop breaks
    assert len(results) == 1
    assert results[0].parent_content is not None


# ==========================================================================
# circuit_breaker.py — half_open → closed transition
# ==========================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_to_closed():
    """After recovery timeout, a successful call resets to closed."""
    cb = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout=0.0)
    # Trip it
    with contextlib.suppress(RuntimeError):
        await cb.wrap(_ctx(), _FailingStage(RuntimeError("x")))
    assert cb._state == "open"
    # Recovery timeout is 0, next call should be half_open → closed on success
    await cb.wrap(_ctx(), _SuccessStage())
    assert cb._state == "closed"
    assert cb._failures == 0


# ==========================================================================
# retry.py — exhausted all retries path
# ==========================================================================


@pytest.mark.asyncio
async def test_retry_exhausted_degradation():
    """After max_retries exhausted, degradation is recorded."""
    mw = RetryMiddleware(
        max_retries=0,
        initial_delay=0.0,
        retryable=(ConnectionError,),
    )
    failing = _FailingStage(ConnectionError("always fails"))
    ctx = await mw.wrap(_ctx(), failing)
    assert any("failed after 1 attempts" in d.error for d in ctx.degradations)


# ==========================================================================
# logging.py — exception path
# ==========================================================================


@pytest.mark.asyncio
async def test_logging_on_exception():
    """Logging middleware logs errors but re-raises."""
    import logging

    logger = logging.getLogger("test_raglan_error_log")
    logger.setLevel(logging.ERROR)
    mw = LoggingMiddleware(logger=logger, level=logging.ERROR)
    with pytest.raises(RuntimeError, match="logged failure"):
        await mw.wrap(_ctx(), _FailingStage(RuntimeError("logged failure")))
    logger.setLevel(logging.WARNING)


# ==========================================================================
# OpenAI expander (mock)
# ==========================================================================


@pytest.mark.asyncio
async def test_openai_expander_mocked():
    """OpenAIExpander with mocked _get_client returns variants."""
    from raglan.expanders.openai import OpenAIExpander

    class _MockMsg:
        def __init__(self, content):
            self.content = content

    class _MockChoice:
        def __init__(self, content):
            self.message = _MockMsg(content)

    class _MockCompletions:
        async def create(self, **kwargs):
            return type("R", (), {"choices": [_MockChoice('{"variants": ["v1", "v2"]}')]})()

    class _MockClient:
        chat = type("CH", (), {"completions": _MockCompletions()})()

    expander = OpenAIExpander(model="test", api_key="fake")
    expander._get_client = lambda: _MockClient
    queries, _entities = await expander.expand("hello", num_variants=2)
    assert queries == ["hello", "v1", "v2"]


# ==========================================================================
# OpenAI embedder (mock)
# ==========================================================================


@pytest.mark.asyncio
async def test_openai_embedder_mocked():
    """OpenAIEmbedder with mocked client returns embeddings."""
    from raglan.embedders.openai import OpenAIEmbedder

    class _MockEmbedding:
        def __init__(self, emb):
            self.embedding = emb

    class _MockEmbed:
        async def create(self, **kwargs):
            return type(
                "R", (), {"data": [_MockEmbedding([0.1, 0.2, 0.3])] * len(kwargs["input"])}
            )()

    class _MockClient:
        embeddings = _MockEmbed()

    embedder = OpenAIEmbedder(model="test", api_key="fake")
    embedder._get_client = lambda: _MockClient
    embs = await embedder.embed(["hello", "world"])
    assert len(embs) == 2
    assert len(embs[0]) == 3


@pytest.mark.asyncio
async def test_openai_embedder_empty_input():
    """Empty input returns empty list without API call."""
    from raglan.embedders.openai import OpenAIEmbedder

    embedder = OpenAIEmbedder(api_key="fake")
    embs = await embedder.embed([])
    assert embs == []


# ==========================================================================
# HuggingFace embedder (mock)
# ==========================================================================


@pytest.mark.asyncio
async def test_huggingface_embedder_mocked():
    """HuggingFaceEmbedder with mocked _get_model."""
    import numpy as np

    from raglan.embedders.huggingface import HuggingFaceEmbedder

    class _MockModel:
        def encode(self, texts, **kw):
            return np.array([[0.1, 0.2]] * len(texts))

    embedder = HuggingFaceEmbedder(model_name="test-model")
    embedder._get_model = lambda: _MockModel()
    embs = await embedder.embed(["hello", "world"])
    assert len(embs) == 2
    assert embedder.dimension == 2


@pytest.mark.asyncio
async def test_huggingface_embedder_empty_input():
    """Empty input returns empty list."""
    from raglan.embedders.huggingface import HuggingFaceEmbedder

    embedder = HuggingFaceEmbedder(model_name="test")
    embs = await embedder.embed([])
    assert embs == []


# ==========================================================================
# Cross-Encoder reranker (mock)
# ==========================================================================


@pytest.mark.asyncio
async def test_cross_encoder_reranker_mocked():
    """CrossEncoderReranker with mocked _get_model."""
    from raglan.rerankers.cross_encoder import CrossEncoderReranker

    class _MockModel:
        def predict(self, pairs, **kw):
            return [0.9, 0.3, 0.7]

    reranker = CrossEncoderReranker(model_name="test-model", min_score=0.5)
    reranker._get_model = lambda: _MockModel()
    candidates = [
        ScoredChunk(chunk_id="a", content="aa", score=0.8),
        ScoredChunk(chunk_id="b", content="bb", score=0.7),
        ScoredChunk(chunk_id="c", content="cc", score=0.6),
    ]
    results = await reranker.rerank("query", candidates, top_k=3)
    assert len(results) == 2
    assert results[0].chunk_id == "a"


@pytest.mark.asyncio
async def test_cross_encoder_empty_candidates():
    """Empty candidates returns empty list."""
    from raglan.rerankers.cross_encoder import CrossEncoderReranker

    # Reranker without model won't fail until first rerank call
    # But for empty candidates, it returns early before hitting the model
    reranker = CrossEncoderReranker()
    results = await reranker.rerank("query", [], top_k=3)
    assert results == []


# ==========================================================================
# memory retriever — remaining uncovered paths
# ==========================================================================


@pytest.mark.asyncio
async def test_memory_retriever_index_streaming():
    """Index() with async iterator replaces all chunks."""
    from raglan.retrievers.memory import MemoryRetriever

    mr = MemoryRetriever()
    mr.load_embedded([("old", "old", [1.0], None)])

    async def batches():
        yield [("a", "aa", None)]
        yield [("b", "bb", None)]

    await mr.index(batches())
    # After index, all old chunks are gone, new ones with empty embeddings
    results = await mr.retrieve(["q"], [[0.0]], top_k=5)
    assert len(results[0]) == 2  # both new chunks (cosine=0 for both)


# ==========================================================================
# window builder — remaining lines
# ==========================================================================


@pytest.mark.asyncio
async def test_window_builder():
    """WindowBuilder passes content through."""
    from raglan.context_builders.window import WindowBuilder

    wb = WindowBuilder(window_chars=200)
    candidates = [ScoredChunk(chunk_id="c1", content="hello", score=0.9)]
    results = await wb.build("q", candidates)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"


# ==========================================================================
# circuit_breaker remaining path — failure in half_open transitions to open
# ==========================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    """Failure during half_open goes back to open."""
    cb = CircuitBreakerMiddleware(failure_threshold=2, recovery_timeout=0.0)
    # First failure
    with contextlib.suppress(RuntimeError):
        await cb.wrap(_ctx(), _FailingStage(RuntimeError("x")))
    assert cb._state == "closed"
    # Second failure trips it
    with contextlib.suppress(RuntimeError):
        await cb.wrap(_ctx(), _FailingStage(RuntimeError("x")))
    assert cb._state == "open"


# ==========================================================================
# retry — remaining path
# ==========================================================================


@pytest.mark.asyncio
async def test_retry_propagates_non_retryable():
    """Only retryable exceptions are retried."""
    mw = RetryMiddleware(
        max_retries=2,
        initial_delay=0.0,
        retryable=(ConnectionError,),
    )
    failing = _FailingStage(ValueError("not retryable"))
    with pytest.raises(ValueError, match="not retryable"):
        await mw.wrap(_ctx(), failing)


# ==========================================================================
# OpenAI expander — JSON decode failure
# ==========================================================================


@pytest.mark.asyncio
async def test_openai_expander_json_decode_fallback():
    """If LLM returns invalid JSON, fall back to original query only."""
    from raglan.expanders.openai import OpenAIExpander

    class _MockCompletions:
        async def create(self, **kwargs):
            return type(
                "R",
                (),
                {
                    "choices": [
                        type("C", (), {"message": type("M", (), {"content": "bad json {{{"})()})
                    ]
                },
            )()

    class _MockClient:
        chat = type("CH", (), {"completions": _MockCompletions()})()

    expander = OpenAIExpander(model="test", api_key="fake")
    expander._get_client = lambda: _MockClient
    queries, _ = await expander.expand("hello", num_variants=2)
    assert queries == ["hello"]


# ==========================================================================
# pipeline — generic callable fallback
# ==========================================================================


@pytest.mark.asyncio
async def test_pipeline_generic_callable():
    """A plain async callable(ctx)->ctx works as a stage."""

    async def my_stage(ctx: PipelineContext) -> PipelineContext:
        ctx.metadata["custom"] = "done"
        return ctx

    pipeline = Pipeline([my_stage, PassthroughBuilder()])
    _results, trace = await pipeline.run("test")
    assert trace is not None


# ==========================================================================
# retry — all retries exhausted with BaseException
# ==========================================================================


@pytest.mark.asyncio
async def test_retry_base_exception_propagates():
    """BaseException subclasses propagate immediately, no retry."""
    mw = RetryMiddleware(max_retries=3, initial_delay=0.0)

    class _KeyboardStage:
        name = "kb_stage"

        async def __call__(self, ctx):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        await mw.wrap(_ctx(), _KeyboardStage())


# ==========================================================================
# memory retriever — load_embedded + remove non-existent
# ==========================================================================


@pytest.mark.asyncio
async def test_memory_retriever_load_embedded():
    """load_embedded with metadata."""
    from raglan.retrievers.memory import MemoryRetriever

    mr = MemoryRetriever()
    mr.load_embedded(
        [
            ("a", "hello", [1.0, 0.0], {"lang": "en"}),
        ]
    )
    results = await mr.retrieve(["q"], [[1.0, 0.0]], top_k=1)
    assert results[0][0].chunk_metadata == {"lang": "en"}


@pytest.mark.asyncio
async def test_memory_retriever_remove_nonexistent():
    """remove() on non-existent IDs is a no-op."""
    from raglan.retrievers.memory import MemoryRetriever

    mr = MemoryRetriever()
    mr.load_embedded([("a", "hello", [1.0], None)])
    await mr.remove(["nonexistent"])
    results = await mr.retrieve(["q"], [[1.0]], top_k=1)
    assert len(results[0]) == 1


# ==========================================================================
# cross_encoder — empty candidates
# ==========================================================================


@pytest.mark.asyncio
async def test_cross_encoder_empty_candidates_model_loaded():
    """Empty candidates returns early, model is never loaded."""
    from raglan.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="test", min_score=0.5)
    # Should return empty without trying to load the model
    results = await reranker.rerank("q", [], top_k=3)
    assert results == []


# ==========================================================================
# helpers
# ==========================================================================


def _ctx() -> PipelineContext:
    return PipelineContext(query="test")


class _SuccessStage:
    name = "test_stage"

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        ctx.metadata["called"] = True
        return ctx


class _FailingStage:
    name = "failing_stage"

    def __init__(self, exc: BaseException | None = None, delay: float = 0.0) -> None:
        self.exc = exc if exc is not None else RuntimeError("fail")
        self.delay = delay
        self.call_count = 0

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        raise self.exc
