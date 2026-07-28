"""Integration tests for QdrantRetriever against a Docker-hosted Qdrant server.

Requires a running Qdrant server (e.g. ``qdrant/qdrant`` Docker image).

Set the connection URL via:
    RAGLAN_QDRANT_URL  (default: http://localhost:6333)
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration


def _get_qdrant_url():
    return os.environ.get("RAGLAN_QDRANT_URL", "http://localhost:6333")


_qdrant_connectivity_cache: tuple[bool, str] | None = None


def _check_connectivity(url: str, retries: int = 10, delay: float = 1.0) -> tuple[bool, str]:
    """Return (True, "") if a Qdrant server is reachable at *url*, else (False, reason).

    Retries several times to allow the container's HTTP server to start.
    """
    global _qdrant_connectivity_cache
    if _qdrant_connectivity_cache is not None:
        return _qdrant_connectivity_cache

    import time
    import urllib.request

    last_error = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(f"{url}/healthz")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    _qdrant_connectivity_cache = (True, "")
                    return _qdrant_connectivity_cache
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            time.sleep(delay)
    _qdrant_connectivity_cache = (False, last_error)
    return _qdrant_connectivity_cache


@pytest.fixture
async def qdrant_retriever():
    """Return a QdrantRetriever connected to the Docker-hosted server."""
    try:
        import qdrant_client  # noqa: F401
    except ImportError:
        pytest.skip("qdrant-client not installed")

    url = _get_qdrant_url()
    ok, reason = _check_connectivity(url)
    if not ok:
        pytest.skip(f"Qdrant server not available at {url}: {reason}")

    from raglan.retrievers.qdrant import QdrantRetriever

    name = f"test_docker_{uuid.uuid4().hex[:8]}"
    r = QdrantRetriever(collection_name=name, url=url, distance_metric="cosine")
    await r._ensure_client()
    yield r
    # Cleanup in the same event loop as the test
    try:
        await r._client.delete_collection(name)
        await r.close()
    except Exception:
        pass


class TestQdrantDockerRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_scored_results(self, qdrant_retriever):
        """Basic retrieval over HTTP returns ranked results."""

        async def gen():
            yield [
                ("c1", "hello world", None, [0.1, 0.2]),
                ("c2", "goodbye world", None, [0.2, 0.1]),
                ("c3", "python programming", None, [0.5, 0.5]),
            ]

        await qdrant_retriever.index(gen())
        results = await qdrant_retriever.retrieve(
            queries=["python"], embeddings=[[0.5, 0.5]], top_k=2
        )
        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0].chunk_id == "c3"

    @pytest.mark.asyncio
    async def test_retrieve_empty_collection(self, qdrant_retriever):
        """Empty collection returns empty results."""
        results = await qdrant_retriever.retrieve(queries=["q"], embeddings=[[0.1, 0.2]], top_k=5)
        assert results[0] == []


class TestQdrantDockerLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, qdrant_retriever):
        """Index → add → remove → verify over HTTP."""

        async def gen():
            yield [("c1", "first", None, [0.1, 0.2])]

        await qdrant_retriever.index(gen())
        await qdrant_retriever.add([("c2", "added", None, [0.9, 0.9])])
        r = await qdrant_retriever.retrieve(queries=["added"], embeddings=[[0.9, 0.9]], top_k=1)
        assert r[0][0].chunk_id == "c2"

        await qdrant_retriever.remove(["c1"])
        r = await qdrant_retriever.retrieve(queries=["first"], embeddings=[[0.1, 0.2]], top_k=5)
        assert "c1" not in {c.chunk_id for c in r[0]}

    @pytest.mark.asyncio
    async def test_index_recreates_collection(self, qdrant_retriever):
        """Re-indexing replaces all data."""

        async def gen1():
            yield [("c1", "batch1", None, [0.1, 0.2])]

        async def gen2():
            yield [("c2", "batch2", None, [0.2, 0.1])]

        await qdrant_retriever.index(gen1())
        await qdrant_retriever.index(gen2())
        results = await qdrant_retriever.retrieve(
            queries=["batch2"], embeddings=[[0.2, 0.1]], top_k=5
        )
        assert results[0][0].chunk_id == "c2"


class TestQdrantDockerFilter:
    @pytest.mark.asyncio
    async def test_eq_filter(self, qdrant_retriever):
        """Equality filter over HTTP."""

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
        assert results[0][0].chunk_id == "c1"
