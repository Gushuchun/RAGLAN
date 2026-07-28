"""Integration tests for ChromaDBRetriever against a Docker-hosted ChromaDB server.

Requires a running ChromaDB server (e.g. ``chromadb/chroma`` Docker image).

Set the connection via environment variables:
    RAGLAN_CHROMADB_HOST  (default: localhost)
    RAGLAN_CHROMADB_PORT  (default: 8000)
"""

from __future__ import annotations

import contextlib
import os

import pytest

pytestmark = pytest.mark.integration


def _get_chromadb_client():
    """Return a HttpClient-based ChromaDB client, or skip if unavailable."""
    try:
        import chromadb
    except ImportError:
        pytest.skip("chromadb not installed")

    host = os.environ.get("RAGLAN_CHROMADB_HOST", "localhost")
    port = os.environ.get("RAGLAN_CHROMADB_PORT", "8000")
    url = f"http://{host}:{port}"

    try:
        client = chromadb.HttpClient(host=host, port=port)
        # Verify connectivity by listing collections
        client.list_collections()
        return client
    except Exception as e:
        pytest.skip(f"ChromaDB server not available at {url}: {e}")


@pytest.fixture
def chroma_retriever():
    """Return a ChromaDBRetriever connected to the Docker-hosted server."""
    import uuid

    from raglan.retrievers.chromadb import ChromaDBRetriever

    client = _get_chromadb_client()
    name = f"test_docker_{uuid.uuid4().hex[:8]}"
    r = ChromaDBRetriever(collection_name=name, client=client)
    yield r
    # Cleanup: delete the test collection
    with contextlib.suppress(Exception):
        client.delete_collection(name)


class TestChromaDBDockerRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_scored_results(self, chroma_retriever):
        """Basic retrieval over HTTP returns correctly scored results."""
        await chroma_retriever.index(
            _batch_gen(
                [
                    ("c1", "hello world", None, [0.1, 0.2]),
                    ("c2", "goodbye world", None, [0.2, 0.1]),
                    ("c3", "python programming", None, [0.5, 0.5]),
                ]
            )
        )
        results = await chroma_retriever.retrieve(
            queries=["python"], embeddings=[[0.5, 0.5]], top_k=2
        )
        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0].chunk_id == "c3"

    @pytest.mark.asyncio
    async def test_retrieve_empty_index(self, chroma_retriever):
        """Empty collection returns empty results."""
        results = await chroma_retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2]], top_k=5)
        assert len(results) == 1
        assert results[0] == []


class TestChromaDBDockerLifecycle:
    @pytest.mark.asyncio
    async def test_index_add_remove(self, chroma_retriever):
        """Full lifecycle over HTTP: index → add → remove → verify."""
        await chroma_retriever.index(_batch_gen([("c1", "first", None, [0.1, 0.2])]))
        await chroma_retriever.add([("c2", "second", None, [0.2, 0.1])])

        r1 = await chroma_retriever.retrieve(queries=["second"], embeddings=[[0.2, 0.1]], top_k=1)
        assert r1[0][0].chunk_id == "c2"

        await chroma_retriever.remove(["c1"])
        r2 = await chroma_retriever.retrieve(queries=["first"], embeddings=[[0.1, 0.2]], top_k=5)
        assert "c1" not in {c.chunk_id for c in r2[0]}

    @pytest.mark.asyncio
    async def test_index_replaces_data(self, chroma_retriever):
        """Re-indexing replaces all data (not appends)."""
        await chroma_retriever.index(_batch_gen([("c1", "batch1", None, [0.1, 0.2])]))
        await chroma_retriever.index(_batch_gen([("c2", "batch2", None, [0.2, 0.1])]))
        results = await chroma_retriever.retrieve(
            queries=["batch2"], embeddings=[[0.2, 0.1]], top_k=5
        )
        assert results[0][0].chunk_id == "c2"


# -- helpers -------------------------------------------------------------------


async def _batch_gen(items):
    yield [(cid, content, meta, emb) for cid, content, meta, emb in items]
