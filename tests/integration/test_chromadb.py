"""Integration tests for ChromaDBRetriever against a real in-memory ChromaDB."""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def chroma_retriever():
    """Return a ChromaDBRetriever backed by an in-memory client.

    Each test gets a unique collection name for isolation.
    """
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    from raglan.retrievers.chromadb import ChromaDBRetriever

    client = chromadb.Client()
    name = f"test_raglan_{uuid.uuid4().hex[:8]}"
    return ChromaDBRetriever(collection_name=name, client=client)


class TestChromaDBRetrieve:
    def test_retrieve_returns_scored_results(self, chroma_retriever):
        """A basic retrieval returns results with correct shape and scoring."""

        # Index some chunks
        async def _index():
            await chroma_retriever.index(
                _batch_gen(
                    [
                        ("c1", "hello world", None, [0.1, 0.2]),
                        ("c2", "goodbye world", None, [0.2, 0.1]),
                        ("c3", "unrelated text", None, [0.5, 0.5]),
                    ]
                )
            )

        asyncio.run(_index())

        results = asyncio.run(
            chroma_retriever.retrieve(queries=["hello"], embeddings=[[0.1, 0.2]], top_k=2)
        )

        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0].chunk_id == "c1"
        assert results[0][0].score > 0

    def test_retrieve_empty_index(self, chroma_retriever):
        """Retrieving from an empty collection returns an empty list."""
        results = asyncio.run(
            chroma_retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2]], top_k=5)
        )
        assert len(results) == 1
        assert results[0] == []

    def test_retrieve_with_filter(self, chroma_retriever):
        """Metadata filters constrain the result set."""
        from raglan.types import Filter

        async def _index():
            await chroma_retriever.index(
                _batch_gen(
                    [
                        ("c1", "doc one", {"status": "active", "lang": "en"}, [0.1, 0.2]),
                        ("c2", "doc two", {"status": "draft", "lang": "en"}, [0.2, 0.1]),
                        ("c3", "doc three", {"status": "active", "lang": "fr"}, [0.3, 0.3]),
                    ]
                )
            )

        asyncio.run(_index())

        results = asyncio.run(
            chroma_retriever.retrieve(
                queries=["doc"],
                embeddings=[[0.1, 0.2]],
                top_k=5,
                filters=[Filter.eq("status", "active")],
            )
        )

        assert len(results[0]) >= 1
        for chunk in results[0]:
            assert chunk.chunk_metadata.get("status") == "active"


class TestChromaDBIndexAddRemove:
    def test_full_index_lifecycle(self, chroma_retriever):
        """Index → add → remove → verify."""

        async def _go():
            # Initial index
            await chroma_retriever.index(_batch_gen([("c1", "first", None, [0.1, 0.2])]))
            results = await chroma_retriever.retrieve(
                queries=["first"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results[0]) == 1

            # Incremental add
            await chroma_retriever.add([("c_new", "unique new text", None, [0.9, 0.9])])
            results = await chroma_retriever.retrieve(
                queries=["unique new text"], embeddings=[[0.9, 0.9]], top_k=1
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c_new"

            # Remove c1 and verify it is gone
            await chroma_retriever.remove(["c1"])
            results = await chroma_retriever.retrieve(
                queries=["first"], embeddings=[[0.1, 0.2]], top_k=5
            )
            removed_ids = {c.chunk_id for c in results[0]}
            assert "c1" not in removed_ids

        asyncio.run(_go())

    def test_index_replaces_all_data(self, chroma_retriever):
        """Index replaces existing data, not appends."""

        async def _go():
            await chroma_retriever.index(_batch_gen([("c1", "batch1", None, [0.1, 0.2])]))
            # Re-index with different data
            await chroma_retriever.index(_batch_gen([("c2", "batch2", None, [0.2, 0.1])]))
            results = await chroma_retriever.retrieve(
                queries=["batch2"], embeddings=[[0.2, 0.1]], top_k=5
            )
            assert len(results[0]) == 1
            assert results[0][0].chunk_id == "c2"

            # Old data should be gone
            results = await chroma_retriever.retrieve(
                queries=["batch1"], embeddings=[[0.1, 0.2]], top_k=5
            )
            assert len(results[0]) == 0 or results[0][0].chunk_id != "c1"

        asyncio.run(_go())


class TestChromaDBDistanceScoring:
    """Verify distance-to-score conversion for cosine, l2, and ip."""

    def test_cosine_distance_scoring(self):
        """Cosine distance (0-2 range) maps to sensible [0,1] scores."""
        from raglan.retrievers.chromadb import _distance_to_score

        assert _distance_to_score(0.0, "cosine") == pytest.approx(1.0)  # identical
        assert _distance_to_score(1.0, "cosine") == pytest.approx(0.5)
        assert _distance_to_score(2.0, "cosine") == pytest.approx(0.0)  # opposite

    def test_l2_distance_scoring(self):
        from raglan.retrievers.chromadb import _distance_to_score

        assert _distance_to_score(0.0, "l2") == pytest.approx(1.0)  # identical
        assert _distance_to_score(1.0, "l2") == pytest.approx(0.5)
        assert _distance_to_score(10.0, "l2") == pytest.approx(1.0 / 11.0)

    def test_ip_distance_scoring(self):
        from raglan.retrievers.chromadb import _distance_to_score

        assert _distance_to_score(1.0, "ip") == pytest.approx(1.0)
        assert _distance_to_score(0.5, "ip") == pytest.approx(0.5)
        assert _distance_to_score(-1.0, "ip") == pytest.approx(0.0)


# -- helpers -------------------------------------------------------------------


async def _batch_gen(items):
    """Yield a single batch (list) of chunk tuples — compatible with AsyncIterator protocol."""
    yield [(cid, content, meta, emb) for cid, content, meta, emb in items]
