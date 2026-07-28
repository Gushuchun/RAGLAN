"""Integration tests for ConfigurablePgvectorRetriever against a Docker-hosted PostgreSQL.

Requires a running pgvector container (e.g. ``pgvector/pgvector:pg17``).

Set the connection string via:
    RAGLAN_PGCONN  (default: postgresql://postgres:postgres@localhost:5432/raglan_test)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.integration

if TYPE_CHECKING:
    import asyncpg
else:
    asyncpg = pytest.importorskip("asyncpg")

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "").replace("gw", "w")
_TABLE = f"kb_chunks_docker{_WORKER}"


@pytest.fixture
async def pg_pool():
    """Return an asyncpg Pool connected to the Docker-hosted PostgreSQL, or skip."""
    cs = os.environ.get(
        "RAGLAN_PGCONN", "postgresql://postgres:postgres@localhost:5432/raglan_test"
    )

    try:
        pool = await asyncpg.create_pool(cs, min_size=1, max_size=2, command_timeout=5)
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    try:
        row = await pool.fetchrow("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        if row is None:
            await pool.close()
            pytest.skip("pgvector extension not available")
    except Exception:
        await pool.close()
        raise

    # Ensure schema
    try:
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:
        if "duplicate key" not in str(e) and "already exists" not in str(e):
            raise
    await pool.execute(f"DROP TABLE IF EXISTS {_TABLE}")
    await pool.execute(f"""
        CREATE TABLE {_TABLE} (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(3),
            metadata JSONB
        )
    """)

    # Seed test data
    rows = [
        ("The quick brown fox", "[0.1,0.2,0.3]", '{"lang":"en","topic":"animals"}'),
        ("Jumped over lazy dogs", "[0.4,0.5,0.6]", '{"lang":"en","topic":"animals"}'),
        ("Machine learning basics", "[0.7,0.8,0.9]", '{"lang":"en","topic":"tech"}'),
    ]
    for content, emb, meta in rows:
        await pool.execute(
            f"INSERT INTO {_TABLE} (content, embedding, metadata) "
            "VALUES ($1, $2::vector, $3::jsonb)",
            content,
            emb,
            meta,
        )

    yield pool

    await pool.execute(f"DROP TABLE IF EXISTS {_TABLE}")
    await pool.close()


@pytest.fixture
async def retriever(pg_pool):
    """Return a ConfigurablePgvectorRetriever connected to the Docker database."""
    from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

    r = ConfigurablePgvectorRetriever(
        table=_TABLE,
        id_column="id",
        content_column="content",
        embedding_column="embedding",
        metadata_column="metadata",
        distance_metric="cosine",
    )
    r._pool = pg_pool
    r._initialised = True
    return r


class TestPgvectorDockerRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_results(self, retriever):
        """Basic retrieval returns ranked results with scores."""
        results = await retriever.retrieve(
            queries=["fox"],
            embeddings=[[0.1, 0.2, 0.3]],
            top_k=2,
        )
        assert len(results) == 1
        assert len(results[0]) >= 1
        assert results[0][0].score > 0

    @pytest.mark.asyncio
    async def test_retrieve_top_k_respected(self, retriever):
        """top_k limits the result count."""
        results = await retriever.retrieve(
            queries=["test"],
            embeddings=[[0.5, 0.5, 0.5]],
            top_k=1,
        )
        assert len(results[0]) == 1

    @pytest.mark.asyncio
    async def test_retrieve_with_filter(self, retriever):
        """Metadata filter via JSONB column works."""
        from raglan.types import Filter

        results = await retriever.retrieve(
            queries=["animals"],
            embeddings=[[0.5, 0.5, 0.5]],
            top_k=5,
            filters=[Filter.eq("topic", "tech")],
        )
        assert len(results[0]) == 1
        assert "Machine learning" in results[0][0].content

    @pytest.mark.asyncio
    async def test_filter_and_combination(self, retriever):
        """AND filter with two conditions."""
        from raglan.types import Filter

        results = await retriever.retrieve(
            queries=["test"],
            embeddings=[[0.5, 0.5, 0.5]],
            top_k=5,
            filters=[Filter.eq("topic", "animals") & Filter.eq("lang", "en")],
        )
        assert len(results[0]) == 2


class TestPgvectorDockerLifecycle:
    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self, retriever):
        """Multiple retrieves reuse the same pool consistently."""
        r1 = await retriever.retrieve(queries=["fox"], embeddings=[[0.1, 0.2, 0.3]], top_k=2)
        r2 = await retriever.retrieve(queries=["fox"], embeddings=[[0.1, 0.2, 0.3]], top_k=2)
        assert r1[0][0].chunk_id == r2[0][0].chunk_id

    @pytest.mark.asyncio
    async def test_close_releases_pool(self, pg_pool):
        """close() releases the pool and resets state."""
        cs = os.environ.get(
            "RAGLAN_PGCONN", "postgresql://postgres:postgres@localhost:5432/raglan_test"
        )
        # Create a dedicated pool so close() doesn't affect the fixture's pool.
        test_pool = await asyncpg.create_pool(cs, min_size=1, max_size=1, command_timeout=5)

        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        r = ConfigurablePgvectorRetriever(
            table=_TABLE,
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            metadata_column="metadata",
        )
        r._pool = test_pool
        r._initialised = True
        await r.close()
        assert r._pool is None
        assert not r._initialised


class TestPgvectorDockerFilter:
    @pytest.mark.asyncio
    async def test_eq_filter(self, retriever):
        """Equality filter returns only matching chunks."""
        from raglan.types import Filter

        results = await retriever.retrieve(
            queries=["animals"],
            embeddings=[[0.5, 0.5, 0.5]],
            top_k=5,
            filters=[Filter.eq("topic", "tech")],
        )
        assert len(results[0]) == 1
        assert "Machine learning" in results[0][0].content

    @pytest.mark.asyncio
    async def test_exists_filter(self, retriever):
        """EXISTS filter returns chunks where the metadata field is present."""
        from raglan.types import Filter

        results = await retriever.retrieve(
            queries=["test"],
            embeddings=[[0.5, 0.5, 0.5]],
            top_k=5,
            filters=[Filter.exists("topic")],
        )
        assert len(results[0]) == 3
