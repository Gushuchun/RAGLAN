"""Tests for provider modules with mocked external dependencies."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# Helpers
# ============================================================================


class _MockEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class _MockEmbedResp:
    def __init__(self, embeddings):
        self.data = [_MockEmbedding(e) for e in embeddings]


class _MockChatMsg:
    def __init__(self, content):
        self.content = content


class _MockChatResp:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": _MockChatMsg(content)})()]


# ============================================================================
# OpenAI Embedder
# ============================================================================


class TestOpenAIEmbedderMocked:
    def test_embed_mocked(self):
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(return_value=_MockEmbedResp([[0.1, 0.2, 0.3]]))

        from raglan.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        embedder._client = mock_client

        import asyncio

        result = asyncio.run(embedder.embed(["hello"]))
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_embed_empty_input(self):
        from raglan.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        import asyncio

        result = asyncio.run(embedder.embed([]))
        assert result == []

    def test_embed_batching(self):
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=lambda model, input, **kw: _MockEmbedResp([[0.1] * 5] * len(input))
        )

        from raglan.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=2)
        embedder._client = mock_client

        import asyncio

        result = asyncio.run(embedder.embed(["a", "b", "c"]))
        assert len(result) == 3
        assert mock_client.embeddings.create.call_count == 2

    def test_dimension_detection(self):
        from raglan.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-large")
        assert embedder.dimension == 3072

    def test_to_dict(self):
        from raglan.embedders.openai import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-small", batch_size=50)
        d = embedder.to_dict()
        assert d["type"] == "openai_embedder"
        assert d["params"]["batch_size"] == 50


# ============================================================================
# OpenAI Expander
# ============================================================================


class TestOpenAIExpanderMocked:
    def test_expand_mocked(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_MockChatResp('{"variants": ["v1", "v2", "v3"]}')
        )

        from raglan.expanders.openai import OpenAIExpander

        expander = OpenAIExpander(model="gpt-4o-mini")
        expander._client = mock_client

        import asyncio

        queries, _entities = asyncio.run(expander.expand("test query", num_variants=3))
        assert queries[0] == "test query"
        assert len(queries) == 4

    def test_expand_original_already_first(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_MockChatResp('{"variants": ["test query", "v1", "v2"]}')
        )

        from raglan.expanders.openai import OpenAIExpander

        expander = OpenAIExpander()
        expander._client = mock_client

        import asyncio

        queries, _ = asyncio.run(expander.expand("test query"))
        assert queries[0] == "test query"

    def test_expand_json_decode_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_MockChatResp("not valid json")
        )

        from raglan.expanders.openai import OpenAIExpander

        expander = OpenAIExpander()
        expander._client = mock_client

        import asyncio

        queries, _ = asyncio.run(expander.expand("test query"))
        assert queries == ["test query"]

    def test_to_dict(self):
        from raglan.expanders.openai import OpenAIExpander

        expander = OpenAIExpander(model="gpt-4o", temperature=0.5)
        d = expander.to_dict()
        assert d["type"] == "openai_expander"
        assert d["params"]["temperature"] == 0.5


# ============================================================================
# HuggingFace Embedder
# ============================================================================


class TestHuggingFaceEmbedderMocked:
    def test_embed_mocked(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))

        from raglan.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(model_name="test-model")
        embedder._model = mock_model

        import asyncio

        result = asyncio.run(embedder.embed(["hello"]))
        assert len(result) == 1
        assert len(result[0]) == 3
        assert embedder.dimension == 3

    def test_embed_empty(self):
        from raglan.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder()
        import asyncio

        result = asyncio.run(embedder.embed([]))
        assert result == []

    def test_to_dict(self):
        from raglan.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder(model_name="BAAI/bge-small", device="cpu")
        d = embedder.to_dict()
        assert d["type"] == "huggingface_embedder"


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================


class TestCrossEncoderRerankerMocked:
    def test_rerank_mocked(self):
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.9, 0.5, 0.2])

        from raglan.rerankers.cross_encoder import CrossEncoderReranker
        from raglan.types import ScoredChunk

        reranker = CrossEncoderReranker(min_score=0.3)
        reranker._model = mock_model

        import asyncio

        candidates = [
            ScoredChunk(chunk_id="1", content="doc1", score=0.8),
            ScoredChunk(chunk_id="2", content="doc2", score=0.7),
            ScoredChunk(chunk_id="3", content="doc3", score=0.6),
        ]
        result = asyncio.run(reranker.rerank("query", candidates, top_k=2, min_score=0.4))
        assert len(result) == 2
        assert result[0].score == 0.9

    def test_rerank_empty_candidates(self):
        from raglan.rerankers.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        import asyncio

        result = asyncio.run(reranker.rerank("q", [], 5))
        assert result == []

    def test_to_dict(self):
        from raglan.rerankers.cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="custom-model", batch_size=16)
        d = reranker.to_dict()
        assert d["type"] == "cross_encoder"
        assert d["params"]["batch_size"] == 16


# ============================================================================
# LiteLLM Expander
# ============================================================================


class TestLiteLLMExpanderMocked:
    def test_expand_mocked(self):
        mock_acompletion = AsyncMock(return_value=_MockChatResp('{"variants": ["v1", "v2"]}'))

        with patch.dict(sys.modules, {"litellm": MagicMock()}):
            sys.modules["litellm"].acompletion = mock_acompletion

            from raglan.expanders.litellm import LiteLLMExpander

            expander = LiteLLMExpander(model="ollama/llama3")

            import asyncio

            queries, _ = asyncio.run(expander.expand("test", num_variants=2))
            assert len(queries) == 3
            assert queries[0] == "test"

    def test_to_dict(self):
        from raglan.expanders.litellm import LiteLLMExpander

        expander = LiteLLMExpander(model="claude-3-haiku-20240307", api_key="sk-test")
        d = expander.to_dict()
        assert d["type"] == "litellm_expander"
        assert d["params"]["api_key"] == "<redacted>"


# ============================================================================
# DashScope Embedder
# ============================================================================


class TestDashScopeEmbedderMocked:
    def test_embed_mocked(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}

        mock_ds = MagicMock()
        mock_ds.TextEmbedding.call = MagicMock(return_value=mock_resp)

        with patch.dict(sys.modules, {"dashscope": mock_ds}):
            from raglan.embedders.dashscope import DashScopeEmbedder

            embedder = DashScopeEmbedder(model="text-embedding-v3")
            import asyncio

            result = asyncio.run(embedder.embed(["hello"]))
            assert len(result) == 1
            assert len(result[0]) == 3

    def test_embed_empty(self):
        from raglan.embedders.dashscope import DashScopeEmbedder

        embedder = DashScopeEmbedder()
        import asyncio

        result = asyncio.run(embedder.embed([]))
        assert result == []

    def test_dimension_defaults(self):
        from raglan.embedders.dashscope import DashScopeEmbedder

        assert DashScopeEmbedder(model="text-embedding-v3").dimension == 1024
        assert DashScopeEmbedder(model="text-embedding-v2").dimension == 1536

    def test_to_dict(self):
        from raglan.embedders.dashscope import DashScopeEmbedder

        embedder = DashScopeEmbedder(model="text-embedding-v3")
        d = embedder.to_dict()
        assert d["type"] == "dashscope_embedder"


# ============================================================================
# Cohere Reranker
# ============================================================================


class TestCohereRerankerMocked:
    def test_rerank_mocked(self):
        mock_result = type(
            "RerankResult",
            (),
            {
                "results": [
                    type("R", (), {"index": 0, "relevance_score": 0.9})(),
                    type("R", (), {"index": 2, "relevance_score": 0.3})(),
                ]
            },
        )()

        mock_client = MagicMock()
        mock_client.rerank = AsyncMock(return_value=mock_result)

        from raglan.rerankers.cohere import CohereReranker
        from raglan.types import ScoredChunk

        reranker = CohereReranker(min_score=0.4)
        reranker._client = mock_client

        import asyncio

        candidates = [
            ScoredChunk(chunk_id="1", content="a", score=0.8),
            ScoredChunk(chunk_id="2", content="b", score=0.7),
            ScoredChunk(chunk_id="3", content="c", score=0.6),
        ]
        result = asyncio.run(reranker.rerank("q", candidates, top_k=3))
        assert len(result) == 1  # Only score >= 0.4

    def test_rerank_empty(self):
        from raglan.rerankers.cohere import CohereReranker

        reranker = CohereReranker()
        import asyncio

        result = asyncio.run(reranker.rerank("q", [], 5))
        assert result == []

    def test_to_dict(self):
        from raglan.rerankers.cohere import CohereReranker

        reranker = CohereReranker(model="rerank-english-v3.0", min_score=0.5)
        d = reranker.to_dict()
        assert d["type"] == "cohere_reranker"


# ============================================================================
# ChromaDB Retriever
# ============================================================================


class TestChromaDBRetrieverMocked:
    def test_init_params(self):
        mock_chroma = MagicMock()
        mock_col = MagicMock()
        mock_chroma.Client.return_value.get_collection.side_effect = [Exception, mock_col]

        with patch.dict(sys.modules, {"chromadb": mock_chroma}):
            # Clear any cached import state
            from raglan.retrievers.chromadb import ChromaDBRetriever

            retriever = ChromaDBRetriever(
                collection_name="test_coll",
                distance_metric="cosine",
            )
            assert retriever.name == "chromadb"
            assert retriever.requires_embeddings is True

    def test_invalid_distance_metric(self):
        with pytest.raises(ValueError, match="distance_metric"):
            from raglan.retrievers.chromadb import ChromaDBRetriever

            ChromaDBRetriever(distance_metric="invalid")

    def test_to_dict(self):
        from raglan.retrievers.chromadb import ChromaDBRetriever

        retriever = ChromaDBRetriever(collection_name="my_coll", persist_directory="/tmp/db")
        d = retriever.to_dict()
        assert d["type"] == "chromadb"


# ============================================================================
# Qdrant Retriever
# ============================================================================


class TestQdrantRetrieverMocked:
    def test_init_params(self):
        from raglan.retrievers.qdrant import QdrantRetriever

        retriever = QdrantRetriever(
            collection_name="test",
            url="http://localhost:6333",
            distance_metric="cosine",
        )
        assert retriever.name == "qdrant"
        assert retriever.requires_embeddings is True

    def test_invalid_distance_metric(self):
        with pytest.raises(ValueError, match="distance_metric"):
            from raglan.retrievers.qdrant import QdrantRetriever

            QdrantRetriever(distance_metric="invalid")

    def test_to_dict(self):
        from raglan.retrievers.qdrant import QdrantRetriever

        retriever = QdrantRetriever(collection_name="coll", url="http://localhost:6333")
        d = retriever.to_dict()
        assert d["type"] == "qdrant"
        assert d["params"]["url"] == "http://localhost:6333"


# ============================================================================
# ConfigurablePgvectorRetriever extra tests
# ============================================================================


class TestPgvectorExtra:
    def test_invalid_distance_metric(self):
        with pytest.raises(ValueError, match="distance_metric"):
            from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

            ConfigurablePgvectorRetriever(table="t", distance_metric="manhattan")

    def test_format_vector(self):
        from raglan.retrievers.configurable_pgvector import _format_vector

        result = _format_vector([1.0, 2.5, 3.14159])
        assert result.startswith("[")
        assert result.endswith("]")

    def test_load_parents_no_parent_col(self):
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        retriever = ConfigurablePgvectorRetriever(table="t")
        import asyncio

        result = asyncio.run(retriever.load_parents(["1", "2"]))
        assert result == {}

    def test_load_parents_empty_ids(self):
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        retriever = ConfigurablePgvectorRetriever(table="t", parent_id_column="pid")
        import asyncio

        result = asyncio.run(retriever.load_parents([]))
        assert result == {}


# ============================================================================
# RetryMiddleware extra coverage
# ============================================================================


class TestRetryMiddlewareExtra:
    def test_invalid_backoff(self):
        from raglan.middleware.retry import RetryMiddleware

        with pytest.raises(ValueError, match="backoff must be"):
            RetryMiddleware(backoff="invalid")

    async def test_non_retryable_propagates(self):
        from raglan.middleware.retry import RetryMiddleware
        from raglan.types import PipelineContext

        mw = RetryMiddleware(max_retries=2, retryable=(ValueError,))
        ctx = PipelineContext(query="test")

        call_count = [0]

        async def _stage(c):
            call_count[0] += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            await mw.wrap(ctx, _stage)
        assert call_count[0] == 1


# ============================================================================
# MemoryRetriever extra coverage
# ============================================================================


class TestMemoryRetrieverExtra:
    async def test_cosine_zero_length(self):
        from raglan.retrievers.memory import _cosine

        assert _cosine([], [1.0]) == 0.0
        assert _cosine([1.0], []) == 0.0

    async def test_cosine_zero_norm(self):
        from raglan.retrievers.memory import _cosine

        assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0
