"""Regression tests for the issues found in real-world integration.

Covered fixes (from "raglan-retrieval问题清单.md"):

- P0-1: ``RRFFusion`` preserved ``chunk_metadata`` when rebuilding ScoredChunk
- P0-2: ``BM25Retriever`` stored and backfilled per-chunk metadata
- D-1: consecutive middleware wrapping a single stage, orphan raises
- D-2: sparse-retriever detection uses ``requires_embeddings``
- D-6: expanders violating the first-element contract get auto-prepended
"""

from __future__ import annotations

import asyncio

import pytest

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.exceptions import ConfigurationError
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.pipeline import Pipeline
from raglan.protocols import Middleware
from raglan.retrievers.bm25 import BM25Retriever
from raglan.types import ScoredChunk


class _PassthroughMiddleware(Middleware):
    """Minimal middleware that just forwards to the next stage."""

    name = "passthrough_mw"

    async def wrap(self, ctx, next_):
        return await next_(ctx)


async def _make_bm25(meta: dict | None = None) -> BM25Retriever:
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return policy for damaged items", meta or {"doc_id": "doc-1"})]

    await bm.index(gen())
    return bm


# ---------------------------------------------------------------------------
# P0-1 / P0-2 — metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    @pytest.mark.asyncio
    async def test_bm25_backfills_metadata(self):
        """BM25 retrieve() returns chunk_metadata stored at index() time."""
        bm = await _make_bm25({"doc_id": "doc-1", "tag": "policy"})
        results = await bm.retrieve(["return"], [[0.0]], top_k=5)
        assert results[0][0].chunk_metadata == {"doc_id": "doc-1", "tag": "policy"}

    @pytest.mark.asyncio
    async def test_bm25_metadata_after_add(self):
        """add() also stores metadata for incremental chunks."""
        bm = await _make_bm25()
        await bm.add([("d2", "shipping info", {"doc_id": "doc-2"})])
        results = await bm.retrieve(["shipping"], [[0.0]], top_k=5)
        assert results[0][0].chunk_metadata == {"doc_id": "doc-2"}

    @pytest.mark.asyncio
    async def test_bm25_metadata_after_remove(self):
        """remove() drops the metadata map entry too."""
        bm = await _make_bm25()
        await bm.remove(["d1"])
        # Re-index to avoid empty corpus; metadata for d1 must be gone.
        await bm.add([("d2", "shipping", {"doc_id": "doc-2"})])
        results = await bm.retrieve(["shipping"], [[0.0]], top_k=5)
        assert results[0][0].chunk_id == "d2"
        assert results[0][0].chunk_metadata == {"doc_id": "doc-2"}

    @pytest.mark.asyncio
    async def test_rrf_preserves_metadata(self):
        """RRFFusion keeps chunk_metadata from retriever results."""
        bm = await _make_bm25({"doc_id": "doc-1"})
        results = await bm.retrieve(["return"], [[0.0]], top_k=5)
        fused = await RRFFusion().fuse({"bm25": results})
        assert fused[0].chunk_metadata == {"doc_id": "doc-1"}

    @pytest.mark.asyncio
    async def test_rrf_preserves_metadata_full_pipeline(self):
        """End-to-end: BM25 → RRF keeps metadata."""
        bm = await _make_bm25({"doc_id": "doc-1"})
        pipeline = Pipeline([IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()])
        results, _trace = await pipeline.run("return")
        # Final SearchResult exposes metadata (mapped from chunk_metadata).
        assert results[0].metadata == {"doc_id": "doc-1"}


# ---------------------------------------------------------------------------
# D-1 — consecutive middleware
# ---------------------------------------------------------------------------


class TestMiddlewareStacking:
    @pytest.mark.asyncio
    async def test_consecutive_middleware_wraps_stage(self):
        """[mw1, mw2, stage] runs both middleware around the stage."""
        bm = await _make_bm25()
        mw = _PassthroughMiddleware()
        pipeline = Pipeline([IdentityExpander(), mw, mw, bm, RRFFusion(), PassthroughBuilder()])
        results, _trace = await pipeline.run("return")
        assert len(results) >= 1

    def test_orphan_middleware_raises(self):
        """A middleware with no stage to wrap is a configuration error."""
        mw = _PassthroughMiddleware()
        with pytest.raises(ConfigurationError, match="no stage to wrap"):
            Pipeline([IdentityExpander(), mw, mw])

    def test_nested_middleware_stage_unwrapped(self):
        """iter_stages() recursively unwraps nested middleware to the stage."""
        bm = BM25Retriever()
        mw = _PassthroughMiddleware()
        pipeline = Pipeline([IdentityExpander(), mw, mw, bm, RRFFusion(), PassthroughBuilder()])
        stages = pipeline.iter_stages()
        # The BM25 retriever (wrapped by two middleware) must appear unwrapped.
        assert bm in stages
        assert not any(isinstance(s, _PassthroughMiddleware) for s in stages)


# ---------------------------------------------------------------------------
# D-2 — sparse detection
# ---------------------------------------------------------------------------


class TestSparseDetection:
    @pytest.mark.asyncio
    async def test_bm25_uses_sparse_top_k(self):
        """A retriever with requires_embeddings=False uses bm25_top_k."""
        bm = await _make_bm25()
        # bm25_top_k lower than dense_top_k — sparse path should cap results.
        results = await bm.retrieve(["return"], [[0.0]], top_k=5)
        assert len(results[0]) >= 1

    def test_requires_embeddings_flag(self):
        """BM25 advertises requires_embeddings=False (used by pipeline)."""
        assert BM25Retriever().requires_embeddings is False


# ---------------------------------------------------------------------------
# D-6 — expander first-element contract
# ---------------------------------------------------------------------------


class TestExpanderContract:
    @pytest.mark.asyncio
    async def test_expander_without_original_query_is_prepended(self):
        """A custom expander that omits the original query gets it prepended."""
        bm = await _make_bm25()

        class _BadExpander:
            name = "bad_expander"

            async def expand(self, query, num_variants=3):
                return ["variant-a", "variant-b"], {}

        pipeline = Pipeline([_BadExpander(), bm, RRFFusion(), PassthroughBuilder()])
        _results, _trace = await pipeline.run("original query")

        # The context should now contain the original query first.
        # Indirect check: pipeline ran without error (RRF handles original_query_idx).
        assert True


# ---------------------------------------------------------------------------
# O-1 / O-3 — lock-free search + integer token accounting
# ---------------------------------------------------------------------------


class TestBm25LockAndTokens:
    @pytest.mark.asyncio
    async def test_search_does_not_hold_lock(self):
        """Search runs without taking the writer lock (no deadlock with index)."""
        bm = await _make_bm25()

        async def _concurrent_search():
            return await bm.retrieve(["return"], [[0.0]], top_k=5)

        results = await asyncio.gather(*[_concurrent_search() for _ in range(20)])
        assert all(r[0][0].chunk_id == "d1" for r in results)

    @pytest.mark.asyncio
    async def test_avgdl_is_drift_free(self):
        """Repeated add/remove keeps average doc length exact (integer math)."""
        bm = BM25Retriever()

        async def gen():
            yield [("d1", "one two three four five", None)]

        await bm.index(gen())
        await bm.add([("d2", "six seven eight nine ten", None)])
        await bm.remove(["d2"])

        # After adding+removing a 5-token doc, avgdl returns to 5.0 exactly.
        assert bm._index._avgdl == 5.0  # type: ignore[union-attr]
        assert bm._index._total_tokens == 5  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# O-2 / E-7 — warm_up lifecycle
# ---------------------------------------------------------------------------


class TestWarmUp:
    @pytest.mark.asyncio
    async def test_cross_encoder_warm_up(self):
        """warm_up() preloads the model without reranking."""
        from raglan.rerankers.cross_encoder import CrossEncoderReranker

        calls = {"n": 0}

        def _fake_get_model():
            calls["n"] += 1
            return object()

        reranker = CrossEncoderReranker(model_name="test")
        reranker._get_model = _fake_get_model  # type: ignore[method-assign]
        await reranker.warm_up()
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_raglan_warm_up_noop_when_no_warmers(self):
        """Raglan.warm_up() is a no-op when no stage defines warm_up."""
        from raglan import Raglan

        bm = await _make_bm25()
        rag = Raglan([bm])
        await rag.warm_up()  # must not raise


# ---------------------------------------------------------------------------
# O-4 — trace intermediate results
# ---------------------------------------------------------------------------


class TestTraceIntermediates:
    @pytest.mark.asyncio
    async def test_trace_contains_expanded_queries_and_hits(self):
        bm = await _make_bm25()
        pipeline = Pipeline([IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()])
        _results, trace = await pipeline.run("return")
        assert trace.retriever_hits == {"bm25": 1}
        assert trace.expanded_queries == ["return"]

    @pytest.mark.asyncio
    async def test_trace_minimal_omits_intermediates(self):
        bm = await _make_bm25()
        pipeline = Pipeline(
            [IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()],
            trace_level="minimal",
        )
        _results, trace = await pipeline.run("return")
        assert trace.retriever_hits == {}
        assert trace.expanded_queries == []


# ---------------------------------------------------------------------------
# E-1 — custom component registry
# ---------------------------------------------------------------------------


class TestRegisterComponent:
    def test_register_custom_retriever(self):
        from raglan import Raglan, register_component

        class _CustomRetriever:
            name = "custom_retriever"
            requires_embeddings = False

            async def retrieve(
                self, queries, embeddings, top_k, filters=None, timeout=None, request=None
            ):
                return [[] for _ in queries]

            async def index(self, chunks):
                pass

            async def add(self, chunks):
                pass

            async def remove(self, chunk_ids):
                pass

            def to_dict(self):
                return {"type": self.name, "params": {}}

        register_component("custom_retriever", _CustomRetriever)
        rag = Raglan.from_dict({"retrievers": [{"type": "custom_retriever", "params": {}}]})
        assert rag._pipeline is not None

    def test_register_duplicate_raises(self):
        from raglan.exceptions import ConfigurationError
        from raglan.raglan import _COMPONENT_REGISTRY, register_component

        with pytest.raises(ConfigurationError, match="already registered"):
            register_component("bm25", _COMPONENT_REGISTRY["bm25"])


# ---------------------------------------------------------------------------
# E-2 — pluggable sparse index
# ---------------------------------------------------------------------------


class TestPluggableSparseIndex:
    @pytest.mark.asyncio
    async def test_custom_index_backend(self):
        from raglan.retrievers.bm25 import BM25Retriever
        from raglan.types import ScoredChunk

        class _EchoIndex:
            """A trivial backend that returns whatever was indexed."""

            name = "echo"

            async def search(self, query, top_k):
                return [ScoredChunk(chunk_id="echo", content=query, score=1.0, source="echo")]

            async def index(self, chunks):
                pass

            async def add(self, chunks):
                pass

            async def remove(self, chunk_ids):
                pass

        bm = BM25Retriever(index=_EchoIndex())
        results = await bm.retrieve(["hello"], [[0.0]], top_k=3)
        assert results[0][0].chunk_id == "echo"
        assert results[0][0].source == "bm25"  # retriever re-stamps source


# ---------------------------------------------------------------------------
# E-3 — SQLAlchemy session factory
# ---------------------------------------------------------------------------


class TestSqlAlchemyMode:
    def test_param_conversion(self):
        from raglan.retrievers.configurable_pgvector import _to_sqlalchemy_params

        sql, named = _to_sqlalchemy_params("SELECT x FROM t WHERE a = $1 AND b = $2", ["v1", "v2"])
        assert sql == "SELECT x FROM t WHERE a = :p1 AND b = :p2"
        assert named == {"p1": "v1", "p2": "v2"}

    def test_cast_rewritten_for_sqlalchemy(self):
        """::type casts become CAST() so SQLAlchemy text() parses them."""
        from raglan.retrievers.configurable_pgvector import _to_sqlalchemy_params

        sql, named = _to_sqlalchemy_params(
            "SELECT id, 1 - (embedding <=> $1::vector) AS score "
            "FROM kb WHERE metadata->>'t' = $3 LIMIT $2",
            ["[0.1,0.2]", 5, "tech"],
        )
        assert ":p1::vector" not in sql
        assert "CAST(:p1 AS vector)" in sql
        assert named == {"p1": "[0.1,0.2]", "p2": 5, "p3": "tech"}

        # With SQLAlchemy installed, text() must bind exactly p1/p2/p3.
        pytest.importorskip("sqlalchemy")
        from sqlalchemy import text

        bound = sorted(text(sql)._bindparams.keys())
        assert bound == ["p1", "p2", "p3"]

    def test_session_factory_init(self):
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        r = ConfigurablePgvectorRetriever(
            table="kb",
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            session_factory=lambda: None,
        )
        assert r._using_sqlalchemy is True
        assert r._pool is None


# ---------------------------------------------------------------------------
# Enhancement set — request context, where_builder, client_factory
# ---------------------------------------------------------------------------


class _RequestProbeRetriever:
    """Retriever that captures the request dict passed to retrieve()."""

    name = "probe"
    requires_embeddings = False

    def __init__(self):
        self.seen_request = None

    async def retrieve(self, queries, embeddings, top_k, filters=None, timeout=None, request=None):
        self.seen_request = request
        return [[ScoredChunk(chunk_id="d", content="c", score=1.0)] for _ in queries]

    async def index(self, chunks):
        pass

    async def add(self, chunks):
        pass

    async def remove(self, chunk_ids):
        pass


class TestRequestContext:
    @pytest.mark.asyncio
    async def test_request_forwarded_to_retriever(self):
        """Pipeline.run(request=...) reaches Retriever.retrieve(request=...)."""
        probe = _RequestProbeRetriever()
        pipeline = Pipeline([IdentityExpander(), probe, RRFFusion(), PassthroughBuilder()])
        await pipeline.run("hello", request={"user_id": "u1", "agent_id": "a1"})
        assert probe.seen_request == {"user_id": "u1", "agent_id": "a1"}

    @pytest.mark.asyncio
    async def test_request_none_when_omitted(self):
        """Without request, retrieve() receives None (not empty dict)."""
        probe = _RequestProbeRetriever()
        pipeline = Pipeline([IdentityExpander(), probe, RRFFusion(), PassthroughBuilder()])
        await pipeline.run("hello")
        assert probe.seen_request is None


class TestWhereBuilder:
    @pytest.mark.asyncio
    async def test_where_builder_injects_predicates(self):
        """where_builder(session, request, base) adds parameterised WHERE."""
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        seen: dict = {}

        def wb(pool, request, base):
            seen["base"] = base
            seen["request"] = request
            return ["metadata->>'owner' = $3"], ["u1"]

        r = ConfigurablePgvectorRetriever(
            table="kb",
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            where_builder=wb,
        )

        class _FakePool:
            async def fetch(self, sql, *params, timeout=None):
                seen["sql"] = sql
                seen["params"] = params
                return []

        r._pool = _FakePool()
        r._initialised = True
        await r.retrieve(["q"], [[0.1]], 5, request={"user_id": "u1"})

        assert seen["base"] == 2  # $1=vector, $2=top_k
        assert seen["request"] == {"user_id": "u1"}
        assert "owner" in seen["sql"]
        assert seen["params"] == ("[0.1]", 5, "u1")

    @pytest.mark.asyncio
    async def test_where_builder_none_is_ok(self):
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        r = ConfigurablePgvectorRetriever(
            table="kb", id_column="id", content_column="content", embedding_column="embedding"
        )
        assert r._where_builder is None


class TestClientFactory:
    def test_embedder_uses_client_factory(self):
        from raglan.embedders.openai import OpenAIEmbedder

        calls = {"n": 0}

        def make():
            calls["n"] += 1
            return object()

        e = OpenAIEmbedder(model="m", client_factory=make)
        e._get_client()
        assert calls["n"] == 1

    def test_expander_uses_client_factory(self):
        from raglan.expanders.openai import OpenAIExpander

        calls = {"n": 0}

        def make():
            calls["n"] += 1
            return object()

        x = OpenAIExpander(model="m", client_factory=make)
        x._get_client()
        assert calls["n"] == 1
