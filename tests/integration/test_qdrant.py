"""Integration tests for QdrantRetriever against an in-memory Qdrant instance."""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def qdrant_retriever():
    """Return a QdrantRetriever backed by an in-memory Qdrant client.

    Each test gets a unique collection name for isolation.
    """
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        pytest.skip("qdrant-client not installed")

    from raglan.retrievers.qdrant import QdrantRetriever

    name = f"test_raglan_{uuid.uuid4().hex[:8]}"
    return QdrantRetriever(collection_name=name, distance_metric="cosine")


class TestQdrantRetrieve:
    def test_retrieve_returns_scored_results(self, qdrant_retriever):
        """Basic retrieval returns results in order of relevance."""

        async def _go():
            async def gen():
                yield [
                    ("c1", "hello world", None, [0.1, 0.2]),
                    ("c2", "goodbye world", None, [0.2, 0.1]),
                    ("c3", "unrelated text", None, [0.5, 0.5]),
                ]

            await qdrant_retriever.index(gen())
            results = await qdrant_retriever.retrieve(
                queries=["hello"], embeddings=[[0.1, 0.2]], top_k=2
            )

            assert len(results) == 1
            assert len(results[0]) == 2
            # First result should be c1 (closest to query embedding)
            assert results[0][0].chunk_id == "c1"
            assert results[0][0].score > 0
            assert results[0][0].content == "hello world"

        asyncio.run(_go())

    def test_retrieve_empty_index(self, qdrant_retriever):
        """Retrieving from an empty collection returns an empty list."""

        async def _go():
            results = await qdrant_retriever.retrieve(
                queries=["q"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results) == 1
            assert results[0] == []

        asyncio.run(_go())

    def test_retrieve_payload_preserved(self, qdrant_retriever):
        """Chunk metadata and parent_chunk_id survive the round trip."""

        async def _go():
            async def gen():
                yield [("c1", "content", {"parent_chunk_id": "p1", "lang": "en"}, [0.1, 0.2])]

            await qdrant_retriever.index(gen())
            results = await qdrant_retriever.retrieve(
                queries=["content"], embeddings=[[0.1, 0.2]], top_k=5
            )

            assert len(results[0]) == 1
            chunk = results[0][0]
            assert chunk.chunk_id == "c1"
            assert chunk.parent_chunk_id == "p1"
            assert chunk.chunk_metadata.get("lang") == "en"

        asyncio.run(_go())


class TestQdrantIndexAddRemove:
    def test_full_lifecycle(self, qdrant_retriever):
        """Index → add → remove → verify."""

        async def _go():
            # Initial index
            async def gen():
                yield [("c1", "first", None, [0.1, 0.2])]

            await qdrant_retriever.index(gen())
            results = await qdrant_retriever.retrieve(
                queries=["first"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results[0]) == 1

            # Incremental add
            await qdrant_retriever.add([("c_new", "unique new text", None, [0.9, 0.9])])
            results = await qdrant_retriever.retrieve(
                queries=["unique new text"], embeddings=[[0.9, 0.9]], top_k=1
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c_new"

            # Remove — deletes only the specified point
            await qdrant_retriever.remove(["c1"])
            results = await qdrant_retriever.retrieve(
                queries=["first"], embeddings=[[0.1, 0.2]], top_k=5
            )
            removed_ids = {c.chunk_id for c in results[0]}
            assert "c1" not in removed_ids

            # c_new should still exist after removing c1
            results = await qdrant_retriever.retrieve(
                queries=["second"], embeddings=[[0.2, 0.1]], top_k=3
            )
            assert len(results[0]) >= 1
            result_ids = {c.chunk_id for c in results[0]}
            assert "c_new" in result_ids

        asyncio.run(_go())

    def test_remove_empty_list_is_noop(self, qdrant_retriever):
        """Removing an empty list does nothing — no error."""

        async def _go():
            async def gen():
                yield [("c1", "data", None, [0.1, 0.2])]

            await qdrant_retriever.index(gen())
            await qdrant_retriever.remove([])  # should not raise
            results = await qdrant_retriever.retrieve(
                queries=["data"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results[0]) >= 1  # data still there

        asyncio.run(_go())

    def test_index_recreates_collection(self, qdrant_retriever):
        """Index replaces all existing data."""

        async def _go():
            async def gen1():
                yield [("c1", "batch1", None, [0.1, 0.2])]

            async def gen2():
                yield [("c2", "batch2", None, [0.2, 0.1])]

            await qdrant_retriever.index(gen1())
            await qdrant_retriever.index(gen2())

            results = await qdrant_retriever.retrieve(
                queries=["batch2"], embeddings=[[0.2, 0.1]], top_k=5
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c2"

            # Old data should be gone
            results = await qdrant_retriever.retrieve(
                queries=["batch1"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results[0]) == 0 or results[0][0].chunk_id != "c1"

        asyncio.run(_go())


class TestQdrantFilter:
    def test_eq_filter(self, qdrant_retriever):
        """Equality filter constrains results."""

        async def _go():
            async def gen():
                yield [
                    ("c1", "text", {"status": "active"}, [0.1, 0.2]),
                    ("c2", "text", {"status": "draft"}, [0.1, 0.2]),
                ]

            from raglan.types import Filter

            await qdrant_retriever.index(gen())
            results = await qdrant_retriever.retrieve(
                queries=["text"],
                embeddings=[[0.1, 0.2]],
                top_k=5,
                filters=[Filter.eq("status", "active")],
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c1"

        asyncio.run(_go())

    def test_and_filter(self, qdrant_retriever):
        """AND combines two leaf conditions."""

        async def _go():
            async def gen():
                yield [
                    ("c1", "text", {"status": "active", "lang": "en"}, [0.1, 0.2]),
                    ("c2", "text", {"status": "active", "lang": "fr"}, [0.1, 0.2]),
                    ("c3", "text", {"status": "draft", "lang": "en"}, [0.1, 0.2]),
                ]

            from raglan.types import Filter

            await qdrant_retriever.index(gen())
            results = await qdrant_retriever.retrieve(
                queries=["text"],
                embeddings=[[0.1, 0.2]],
                top_k=5,
                filters=[Filter.eq("status", "active") & Filter.eq("lang", "en")],
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c1"

        asyncio.run(_go())
