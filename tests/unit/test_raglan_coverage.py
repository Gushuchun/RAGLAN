"""Additional tests for Raglan facade, Builder, and Pipeline edge cases."""

from __future__ import annotations

import pytest

from raglan import (
    ConfigurationError,
    LoggingMetricsCollector,
    NoOpMetricsCollector,
    Pipeline,
    Raglan,
)
from raglan.context_builders import ParentExpander, PassthroughBuilder
from raglan.embedders import OpenAIEmbedder
from raglan.expanders import IdentityExpander, OpenAIExpander
from raglan.fusion import RoundRobinFusion, RRFFusion, WeightedFusion
from raglan.retrievers import BM25Retriever, MemoryRetriever
from raglan.types import PipelineContext, ScoredChunk


class TestRaglanFromDict:
    """Comprehensive from_dict/export_config round-trip tests."""

    async def test_from_dict_full_config(self):
        config = {
            "fallback_mode": "degrade",
            "expander": {"type": "identity", "params": {}},
            "embedder": {"type": "openai_embedder", "params": {"model": "text-embedding-3-small"}},
            "retrievers": [
                {"type": "bm25", "params": {"k1": 1.2, "b": 0.5}},
                {"type": "memory", "params": {}},
            ],
            "fusion": {"type": "rrf", "params": {"k": 60}},
            "reranker": {"type": "cross_encoder", "params": {"min_score": 0.3}},
            "context_builder": {"type": "passthrough", "params": {}},
        }
        rag = Raglan.from_dict(config)
        exported = rag.export_config()
        assert exported["fallback_mode"] == "degrade"
        assert len(exported["retrievers"]) == 2

    async def test_from_dict_minimal(self):
        config = {"retrievers": [{"type": "bm25", "params": {}}]}
        rag = Raglan.from_dict(config)
        assert rag is not None
        exported = rag.export_config()
        assert "retrievers" in exported

    async def test_from_dict_with_weighted_fusion(self):
        config = {
            "retrievers": [{"type": "bm25", "params": {}}],
            "fusion": {"type": "weighted", "params": {"weights": {"bm25": 0.5}}},
        }
        rag = Raglan.from_dict(config)
        assert rag is not None

    async def test_from_dict_with_round_robin_fusion(self):
        config = {
            "retrievers": [{"type": "bm25", "params": {}}],
            "fusion": {"type": "round_robin", "params": {}},
        }
        rag = Raglan.from_dict(config)
        assert rag is not None

    async def test_from_dict_with_all_fusion_types(self):
        for fusion_type in ["rrf", "weighted", "round_robin"]:
            config = {
                "retrievers": [{"type": "bm25", "params": {}}],
                "fusion": {"type": fusion_type, "params": {}},
            }
            rag = Raglan.from_dict(config)
            assert rag is not None

    async def test_from_dict_unknown_type_raises(self):
        config = {
            "retrievers": [{"type": "nonexistent_retriever", "params": {}}],
        }
        with pytest.raises(ConfigurationError, match="Unknown component type"):
            Raglan.from_dict(config)

    async def test_from_dict_missing_type_key(self):
        config = {"retrievers": [{"params": {}}]}
        with pytest.raises(ConfigurationError, match="must include a 'type' key"):
            Raglan.from_dict(config)

    async def test_from_dict_strict_mode(self):
        config = {
            "fallback_mode": "strict",
            "retrievers": [{"type": "bm25", "params": {}}],
        }
        rag = Raglan.from_dict(config)
        assert rag.export_config()["fallback_mode"] == "strict"

    async def test_to_dict_alias(self):
        bm = BM25Retriever()
        rag = Raglan.builder().with_retrievers([bm]).build()
        assert rag.to_dict() == rag.export_config()

    async def test_export_config_with_expander(self):
        bm = BM25Retriever()
        rag = (
            Raglan.builder()
            .with_expander(OpenAIExpander(model="test"))
            .with_retrievers([bm])
            .build()
        )
        config = rag.export_config()
        assert "expander" in config
        assert config["expander"]["type"] == "openai_expander"

    async def test_export_config_with_embedder(self):
        rag = (
            Raglan.builder()
            .with_embedder(OpenAIEmbedder(model="test"))
            .with_retrievers([MemoryRetriever()])
            .build()
        )
        config = rag.export_config()
        assert "embedder" in config

    async def test_export_config_with_reranker(self):
        from raglan.rerankers import CrossEncoderReranker

        rag = (
            Raglan.builder()
            .with_retrievers([BM25Retriever()])
            .with_reranker(CrossEncoderReranker(min_score=0.5))
            .build()
        )
        config = rag.export_config()
        assert "reranker" in config

    async def test_export_config_with_context_builder(self):
        async def loader(ids):
            return {}

        rag = (
            Raglan.builder()
            .with_retrievers([BM25Retriever()])
            .with_context_builder(ParentExpander(loader=loader))
            .build()
        )
        config = rag.export_config()
        assert "context_builder" in config

    async def test_export_config_omits_defaults(self):
        rag = Raglan.builder().with_retrievers([BM25Retriever()]).build()
        config = rag.export_config()
        # IdentityExpander and PassthroughBuilder are defaults, should be omitted
        assert "expander" not in config
        assert "context_builder" not in config


class TestRaglanBuilder:
    """Additional Builder edge cases."""

    def test_builder_with_expander_method(self):
        builder = Raglan.builder().with_expander(OpenAIExpander(model="test"))
        result = builder.with_retrievers([BM25Retriever()]).with_fallback_mode("strict")
        assert result is builder

    def test_builder_with_metrics_collector(self):
        rag = (
            Raglan.builder()
            .with_retrievers([BM25Retriever()])
            .with_metrics_collector(NoOpMetricsCollector())
            .build()
        )
        assert rag is not None

    def test_builder_embedder_before_validates(self):
        rag = (
            Raglan.builder()
            .with_embedder(OpenAIEmbedder(model="test"))
            .with_retrievers([MemoryRetriever()])
            .build()
        )
        assert rag is not None


class TestPipelineEdgeCases:
    """Pipeline edge cases for metrics and degradation."""

    @pytest.mark.asyncio
    async def test_pipeline_with_metrics_collector(self):
        collector = LoggingMetricsCollector()
        bm = BM25Retriever()

        async def gen():
            yield [("d1", "test", None)]

        await bm.index(gen())
        pipeline = Pipeline(
            [IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()],
            metrics_collector=collector,
        )
        results, trace = await pipeline.run("test")
        assert len(results) == 1
        assert trace is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_timeout_success(self):
        bm = BM25Retriever()

        async def gen():
            yield [("d1", "test", None)]

        await bm.index(gen())
        pipeline = Pipeline(
            [IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()],
        )
        results, _trace = await pipeline.run("test", timeout=10.0)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_pipeline_search_options_top_k(self):
        bm = BM25Retriever()

        async def gen():
            yield [
                ("d1", "first document about search", None),
                ("d2", "second document about query", None),
            ]

        await bm.index(gen())
        rag = Raglan.builder().with_retrievers([bm]).build()
        results, _trace = await rag.search("search", top_k=1)
        assert len(results) == 1


class TestFusionEdgeCases:
    """Additional fusion edge cases for coverage."""

    @pytest.mark.asyncio
    async def test_round_robin_parent_dedup(self):
        fusion = RoundRobinFusion()
        chunks = [
            ScoredChunk(chunk_id="c1", content="a", score=0.9, parent_chunk_id="p1"),
            ScoredChunk(chunk_id="c2", content="b", score=0.8, parent_chunk_id="p1"),
        ]
        results = {"ret1": [[chunks[0]]], "ret2": [[chunks[1]]]}
        fused = await fusion.fuse(results)
        assert len(fused) == 1  # deduplicated by parent

    @pytest.mark.asyncio
    async def test_round_robin_empty(self):
        fusion = RoundRobinFusion()
        fused = await fusion.fuse({})
        assert fused == []

    @pytest.mark.asyncio
    async def test_weighted_fusion_custom_weights(self):
        fusion = WeightedFusion(weights={"a": 0.7, "b": 0.3})
        results = {
            "a": [[ScoredChunk(chunk_id="c1", content="x", score=0.9)]],
            "b": [[ScoredChunk(chunk_id="c2", content="y", score=0.5)]],
        }
        fused = await fusion.fuse(results)
        assert len(fused) == 2

    @pytest.mark.asyncio
    async def test_weighted_fusion_equal_weights_default(self):
        fusion = WeightedFusion()
        results = {
            "a": [[ScoredChunk(chunk_id="c1", content="x", score=0.9)]],
        }
        fused = await fusion.fuse(results)
        assert len(fused) == 1

    def test_to_dict_fusion_types(self):
        assert "type" in RRFFusion().to_dict()
        assert "type" in WeightedFusion().to_dict()
        assert "type" in RoundRobinFusion().to_dict()

    def test_retry_budget_to_dict_not_needed(self):
        """Protocol components with no to_dict still serialize via _serialize fallback."""
        from raglan.raglan import _serialize

        result = _serialize(WeightedFusion())
        assert result["type"] == "weighted"


class TestEdgeCases:
    """Miscellaneous edge cases for full coverage."""

    def test_filter_operator_overloads(self):
        from raglan.types import Filter, Op

        f1 = Filter.eq("status", "active")
        f2 = Filter.eq("dept", "sales")
        combined = f1 & f2
        assert combined.op == Op.AND
        combined_or = f1 | f2
        assert combined_or.op == Op.OR

    def test_filter_bool_forbidden(self):
        from raglan.types import Filter

        f = Filter.eq("x", 1)
        with pytest.raises(TypeError, match="cannot be evaluated as booleans"):
            bool(f)

    def test_pipeline_context_defaults(self):
        ctx = PipelineContext()
        assert ctx.query == ""
        assert ctx.expanded_queries == []
        assert ctx.embeddings == []
        assert ctx.fused_candidates == []
        assert ctx.final_results == []

    def test_search_sync_wrapper(self):
        """search_sync must run outside the event loop."""
        import asyncio as _asyncio

        async def _setup():
            bm = BM25Retriever()

            async def gen():
                yield [("d1", "hello world", None)]

            await bm.index(gen())
            rag = Raglan.builder().with_retrievers([bm]).build()
            return rag

        rag = _asyncio.run(_setup())
        results, _trace = rag.search_sync("hello")
        assert len(results) == 1

    def test_trace_degraded_property(self):
        from raglan.types import StageDegradation, Trace

        trace = Trace(query="test")
        assert trace.degraded is False
        trace.degradations.append(StageDegradation(stage="x", error="e"))
        assert trace.degraded is True
        assert trace.degraded_stage_names == ["x"]

    def test_configuration_error_hierarchy(self):
        from raglan.exceptions import ConfigurationError, RaglanError

        assert issubclass(ConfigurationError, RaglanError)
