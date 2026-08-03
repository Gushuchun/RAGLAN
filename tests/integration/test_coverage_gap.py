"""Targeted tests to close remaining coverage gaps."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# ChromaDB mocked retrieve/index/add
# ============================================================================


class TestChromaDBFullMock:
    def _make_mock_collection(self, ids, documents, distances, metadatas):
        """Build a mock ChromaDB collection that supports query/get/add/delete."""
        col = MagicMock()
        col.query.return_value = {
            "ids": [ids],
            "documents": [documents],
            "distances": [distances],
            "metadatas": [metadatas],
        }
        col.get.return_value = {"ids": []}
        col.add.return_value = None
        col.delete.return_value = None
        return col

    def test_retrieve_with_mock(self):
        mock_chroma = MagicMock()
        mock_col = self._make_mock_collection(
            ids=["d1", "d2"],
            documents=["doc one", "doc two"],
            distances=[0.1, 0.3],
            metadatas=[{"k": "v"}, {}],
        )
        mock_chroma.Client.return_value.get_collection.return_value = mock_col

        with patch.dict(sys.modules, {"chromadb": mock_chroma}):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = mock_col

            import asyncio

            results = asyncio.run(
                retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2]], top_k=5)
            )
            assert len(results) == 1
            assert len(results[0]) == 2
            assert results[0][0].chunk_id == "d1"

    def test_retrieve_filter_passthrough(self):
        mock_col = self._make_mock_collection(
            ids=["d1"], documents=["doc"], distances=[0.1], metadatas=[{}]
        )

        with patch.dict(sys.modules, {"chromadb": MagicMock()}):
            from raglan.retrievers.chromadb import ChromaDBRetriever
            from raglan.types import Filter

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = mock_col

            import asyncio

            results = asyncio.run(
                retriever.retrieve(
                    queries=["q"],
                    embeddings=[[0.1]],
                    top_k=5,
                    filters=[Filter.eq("status", "active")],
                )
            )
            assert len(results[0]) == 1

    def test_index_streaming(self):
        mock_col = self._make_mock_collection([], [], [], [])
        mock_chroma = MagicMock()
        mock_chroma.Client.return_value.get_collection.return_value = mock_col

        with patch.dict(sys.modules, {"chromadb": mock_chroma}):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = mock_col

            import asyncio

            async def batch_gen():
                yield [("c1", "hello world", {"lang": "en"}, [0.1, 0.2])]
                yield [("c2", "foo bar", None)]

            asyncio.run(retriever.index(batch_gen()))
            assert mock_col.add.call_count == 2

    def test_add_chunks(self):
        mock_col = self._make_mock_collection([], [], [], [])

        with patch.dict(sys.modules, {"chromadb": MagicMock()}):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = mock_col

            import asyncio

            asyncio.run(retriever.add([("c1", "text", {"lang": "en"}, [0.1])]))
            mock_col.add.assert_called_once()

    def test_remove_chunks(self):
        mock_col = self._make_mock_collection([], [], [], [])

        with patch.dict(sys.modules, {"chromadb": MagicMock()}):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = mock_col

            import asyncio

            asyncio.run(retriever.remove(["c1", "c2"]))
            mock_col.delete.assert_called_once_with(ids=["c1", "c2"])


# ============================================================================
# Qdrant mocked retrieve/index
# ============================================================================


class TestQdrantFullMock:
    def test_retrieve_with_mock(self):
        mock_qdrant = MagicMock()
        mock_qdrant.models = MagicMock()
        mock_qdrant.AsyncQdrantClient = MagicMock()

        mock_point = MagicMock()
        mock_point.id = "pt1"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "hello",
            "chunk_id": "c1",
            "parent_chunk_id": "p1",
            "metadata": {},
        }

        mock_resp = MagicMock()
        mock_resp.points = [mock_point]

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=mock_resp)
        mock_client.get_collection = AsyncMock()
        mock_qdrant.AsyncQdrantClient.return_value = mock_client

        with patch.dict(sys.modules, {"qdrant_client": mock_qdrant}):
            from raglan.retrievers.qdrant import QdrantRetriever

            retriever = QdrantRetriever(collection_name="test", url="http://localhost:6333")
            retriever._client = mock_client
            retriever._initialised = True

            import asyncio

            results = asyncio.run(
                retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2]], top_k=5)
            )
            assert len(results) == 1
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c1"

    def test_index_with_mock(self):
        mock_qdrant = MagicMock()
        mock_models = MagicMock()
        mock_models.Distance = MagicMock()
        mock_models.VectorParams = MagicMock()
        mock_qdrant.models = mock_models

        with patch.dict(
            sys.modules, {"qdrant_client": mock_qdrant, "qdrant_client.models": mock_models}
        ):
            from raglan.retrievers.qdrant import QdrantRetriever

            mock_client = MagicMock()
            mock_client.delete_collection = AsyncMock()
            mock_client.get_collection = AsyncMock()
            mock_client.create_collection = AsyncMock()
            mock_client.upsert = AsyncMock()

            retriever = QdrantRetriever(collection_name="test", url="http://localhost:6333")
            retriever._client = mock_client
            retriever._initialised = True

            import asyncio

            async def batch_gen():
                yield [("c1", "hello world", {"parent_chunk_id": "p1"}, [0.1, 0.2])]

            asyncio.run(retriever.index(batch_gen()))
            mock_client.upsert.assert_called()

    def test_add_and_remove(self):
        mock_qdrant = MagicMock()
        mock_models = MagicMock()
        mock_qdrant.models = mock_models

        with patch.dict(
            sys.modules, {"qdrant_client": mock_qdrant, "qdrant_client.models": mock_models}
        ):
            from raglan.retrievers.qdrant import QdrantRetriever

            mock_client = MagicMock()
            mock_client.get_collection = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_client.delete = AsyncMock()

            retriever = QdrantRetriever(collection_name="test", url="http://localhost:6333")
            retriever._client = mock_client
            retriever._initialised = True

            import asyncio

            asyncio.run(retriever.add([("c1", "text", None)]))
            mock_client.upsert.assert_called()

            asyncio.run(retriever.remove(["c1"]))
            mock_client.delete.assert_called()


# ============================================================================
# Pgvector mocked retrieve
# ============================================================================


class TestPgvectorFullMock:
    def test_retrieve_with_mock_pool(self):
        mock_asyncpg = MagicMock()
        mock_pool = MagicMock()

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, idx: {0: "d1", 1: "content", 2: "p1", 3: 0.95}[idx]
        mock_pool.fetch = AsyncMock(return_value=[mock_row])

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

            retriever = ConfigurablePgvectorRetriever(table="t", parent_id_column="parent_id")
            retriever._pool = mock_pool
            retriever._initialised = True

            import asyncio

            results = asyncio.run(
                retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2, 0.3]], top_k=5)
            )
            assert len(results) == 1
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "d1"
            assert results[0][0].parent_chunk_id == "p1"

    def test_retrieve_no_parent_col(self):
        mock_asyncpg = MagicMock()
        mock_pool = MagicMock()

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, idx: {0: "d1", 1: "content", 2: "d1", 3: 0.95}[idx]
        mock_pool.fetch = AsyncMock(return_value=[mock_row])

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

            retriever = ConfigurablePgvectorRetriever(table="t")
            retriever._pool = mock_pool
            retriever._initialised = True

            import asyncio

            results = asyncio.run(retriever.retrieve(queries=["q"], embeddings=[[0.1]], top_k=5))
            assert len(results[0]) == 1

    def test_load_parents_with_data(self):
        mock_asyncpg = MagicMock()
        mock_pool = MagicMock()
        mock_row = MagicMock()

        def _getitem(self, key):
            # Support both positional access (asyncpg Record style, row[0])
            # and column-name access (row["child_id"]).
            if isinstance(key, int):
                return ["c1", "pc"][key]
            return {"child_id": "c1", "parent_content": "pc"}[key]

        mock_row.__getitem__ = _getitem
        mock_pool.fetch = AsyncMock(return_value=[mock_row])

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

            retriever = ConfigurablePgvectorRetriever(table="t", parent_id_column="parent_id")
            retriever._pool = mock_pool
            retriever._initialised = True

            import asyncio

            result = asyncio.run(retriever.load_parents(["c1"]))
            assert result == {"c1": "pc"}


# ============================================================================
# ImportError paths for optional deps
# ============================================================================


class TestImportErrorPaths:
    def test_openai_embedder_import_error(self):
        with patch.dict(sys.modules, {"openai": None}):
            # Force reimport

            if "raglan.embedders.openai" in sys.modules:
                del sys.modules["raglan.embedders.openai"]

            from raglan.embedders.openai import OpenAIEmbedder

            embedder = OpenAIEmbedder()
            embedder._client = None
            # Remove pkg from sys.modules to trigger ImportError
            with (
                patch.dict(sys.modules, {"openai": None}),
                pytest.raises(ImportError, match="requires the 'openai' package"),
            ):
                import asyncio

                asyncio.run(embedder.embed(["test"]))

    def test_cohere_reranker_import_error(self):
        with patch.dict(sys.modules, {"cohere": None}):
            from raglan.rerankers.cohere import CohereReranker
            from raglan.types import ScoredChunk

            reranker = CohereReranker()
            reranker._client = None
            with (
                patch.dict(sys.modules, {"cohere": None}),
                pytest.raises(ImportError, match="requires the 'cohere' package"),
            ):
                import asyncio

                asyncio.run(
                    reranker.rerank("q", [ScoredChunk(chunk_id="1", content="c", score=0.5)], 2)
                )

    def test_huggingface_embedder_import_error(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from raglan.embedders.huggingface import HuggingFaceEmbedder

            embedder = HuggingFaceEmbedder()
            embedder._model = None
            with (
                patch.dict(sys.modules, {"sentence_transformers": None}),
                pytest.raises(ImportError, match="requires the 'sentence_transformers' package"),
            ):
                import asyncio

                asyncio.run(embedder.embed(["test"]))

    def test_cross_encoder_import_error(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from raglan.rerankers.cross_encoder import CrossEncoderReranker
            from raglan.types import ScoredChunk

            reranker = CrossEncoderReranker()
            reranker._model = None
            with (
                patch.dict(sys.modules, {"sentence_transformers": None}),
                pytest.raises(ImportError, match="requires the 'sentence_transformers' package"),
            ):
                import asyncio

                asyncio.run(
                    reranker.rerank("q", [ScoredChunk(chunk_id="1", content="c", score=0.5)], 2)
                )

    def test_litellm_expander_import_error(self):
        with patch.dict(sys.modules, {"litellm": None}):
            from raglan.expanders.litellm import LiteLLMExpander

            expander = LiteLLMExpander()
            with (
                patch.dict(sys.modules, {"litellm": None}),
                pytest.raises(ImportError, match="requires litellm"),
            ):
                import asyncio

                asyncio.run(expander.expand("test"))

    def test_dashscope_embedder_import_error(self):
        import asyncio

        from raglan.embedders.dashscope import DashScopeEmbedder

        embedder = DashScopeEmbedder()
        # dashscope is not installed, so embed() will raise ImportError
        with pytest.raises((ImportError, RuntimeError)):
            asyncio.run(embedder.embed(["test"]))

    def test_pgvector_retriever_import_error(self):
        """retrieve() raises when pool is not initialised and connect fails."""
        import asyncio

        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        retriever = ConfigurablePgvectorRetriever(
            connection_string="postgresql://nohost.invalid:54321/test",
            table="t",
            id_column="id",
            content_column="content",
            embedding_column="embedding",
        )
        retriever._pool = None
        # _ensure_pool() will fail because the host is unreachable.
        # We accept any asyncpg/OAuth error since the exception type varies.
        with pytest.raises(Exception):  # noqa: B017
            asyncio.run(
                retriever.retrieve(
                    queries=["q"],
                    embeddings=[[0.1]],
                    top_k=5,
                    timeout=0.5,
                )
            )

    def test_chromadb_import_error(self):
        with patch.dict(sys.modules, {"chromadb": None}):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(collection_name="test")
            retriever._collection = None
            with (
                patch.dict(sys.modules, {"chromadb": None}),
                pytest.raises(ImportError, match="requires the 'chromadb' package"),
            ):
                import asyncio

                asyncio.run(retriever.retrieve(queries=["q"], embeddings=[[0.1]], top_k=5))

    def test_qdrant_import_error(self):
        with patch.dict(sys.modules, {"qdrant_client": None}):
            from raglan.retrievers.qdrant import QdrantRetriever

            retriever = QdrantRetriever(collection_name="test", url="http://localhost:6333")
            retriever._client = None
            retriever._initialised = False
            with (
                patch.dict(sys.modules, {"qdrant_client": None}),
                pytest.raises(ImportError, match="requires the 'qdrant_client' package"),
            ):
                import asyncio

                asyncio.run(retriever.retrieve(queries=["q"], embeddings=[[0.1]], top_k=5))


# ============================================================================
# Pipeline stage dispatch edge cases
# ============================================================================


class TestPipelineDispatchExtra:
    @pytest.mark.asyncio
    async def test_pipeline_with_embedder_stage(self):
        """Pipeline with a retriever that requires embeddings but has no queries."""
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=lambda model, input, **kw: type(
                "Resp", (), {"data": [type("D", (), {"embedding": [0.1] * 3})() for _ in input]}
            )()
        )

        from raglan.context_builders.passthrough import PassthroughBuilder
        from raglan.embedders.openai import OpenAIEmbedder
        from raglan.expanders.identity import IdentityExpander
        from raglan.fusion.rrf import RRFFusion
        from raglan.pipeline import Pipeline
        from raglan.retrievers import MemoryRetriever

        embedder = OpenAIEmbedder(model="test")
        embedder._client = mock_client

        retriever = MemoryRetriever()
        retriever._chunks = [("d1", "hello", [0.1] * 3, {})]

        pipeline = Pipeline(
            [IdentityExpander(), embedder, retriever, RRFFusion(), PassthroughBuilder()]
        )
        _results, trace = await pipeline.run("test")
        # Should not crash even with single embedder+retriever
        assert trace is not None

    @pytest.mark.asyncio
    async def test_pipeline_reranker_without_fusion_results(self):
        """Reranker falls back to retriever_results when fused_candidates is empty."""
        from raglan.context_builders.passthrough import PassthroughBuilder
        from raglan.expanders.identity import IdentityExpander
        from raglan.pipeline import Pipeline
        from raglan.rerankers.cross_encoder import CrossEncoderReranker
        from raglan.retrievers import BM25Retriever

        bm = BM25Retriever()

        async def gen():
            yield [("d1", "target content here", None)]

        await bm.index(gen())

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9])

        reranker = CrossEncoderReranker(min_score=0.0)
        reranker._model = mock_model

        # No fusion stage, reranker uses retriever_results directly
        pipeline = Pipeline([IdentityExpander(), bm, reranker, PassthroughBuilder()])
        results, _trace = await pipeline.run("target")
        assert len(results) >= 1


# ============================================================================
# RoundRobin and Weighted edge cases
# ============================================================================


class TestFusionRemaining:
    @pytest.mark.asyncio
    async def test_round_robin_to_dict(self):
        from raglan.fusion.round_robin import RoundRobinFusion

        fusion = RoundRobinFusion()
        d = fusion.to_dict()
        assert d["type"] == "round_robin"

    @pytest.mark.asyncio
    async def test_weighted_fusion_to_dict(self):
        from raglan.fusion.weighted import WeightedFusion

        fusion = WeightedFusion(weights={"a": 0.8})
        d = fusion.to_dict()
        assert d["type"] == "weighted"
        assert d["params"]["weights"] == {"a": 0.8}


# ============================================================================
# IdentityExpander and PassthroughBuilder serde
# ============================================================================


class TestIdentityPassthrough:
    def test_identity_is_query_expander(self):
        from raglan.expanders.identity import IdentityExpander
        from raglan.protocols import QueryExpander

        assert isinstance(IdentityExpander(), QueryExpander)

    def test_passthrough_is_context_builder(self):
        from raglan.context_builders.passthrough import PassthroughBuilder
        from raglan.protocols import ContextBuilder

        assert isinstance(PassthroughBuilder(), ContextBuilder)
