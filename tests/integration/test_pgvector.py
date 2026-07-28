"""Integration tests for ConfigurablePgvectorRetriever against a real Postgres + pgvector.

Connection strategy (tried in order):
1. ``RAGLAN_PGCONN`` env var (for CI / Docker / custom)
2. ``postgresql://postgres:postgres@localhost:5432/raglan_test`` (Docker default)
3. ``postgresql://postgres:123456@localhost:5432/raglan_test`` (local dev)

If none are reachable the tests are skipped.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.integration
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import asyncpg
else:
    asyncpg = pytest.importorskip("asyncpg")

_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "").replace("gw", "w")
_TABLE = f"kb_chunks{_WORKER}"

# Connection strings to try, in order
_CANDIDATE_CONNS = [
    os.environ.get("RAGLAN_PGCONN"),
    "postgresql://postgres:postgres@localhost:5432/raglan_test",
    "postgresql://postgres:123456@localhost:5432/raglan_test",
]


async def _try_connect() -> tuple[asyncpg.Pool | None, str]:
    """Try each candidate connection string.  Return (pool, "") on success, or (None, reason)."""
    last_error = "no candidates configured"
    for cs in _CANDIDATE_CONNS:
        if cs is None:
            continue
        try:
            pool = await asyncpg.create_pool(cs, min_size=1, max_size=2, command_timeout=10)
            # Verify pgvector is available
            row = await pool.fetchrow("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            if row is not None:
                logger.info(
                    "Connected to PostgreSQL via %s", cs.split("@")[-1] if "@" in cs else cs
                )
                return pool, ""
            await pool.close()
            last_error = "pgvector extension not available"
        except Exception as exc:
            target = cs.split("@")[-1] if "@" in cs else cs
            last_error = f"{target}: {exc}"
            logger.info("PostgreSQL connection attempt failed: %s", last_error)
    return None, last_error


async def _ensure_schema(pool):
    """Create the test table and extension if needed."""
    try:
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as e:
        if "duplicate key" not in str(e) and "already exists" not in str(e):
            raise
    await pool.execute(f"DROP TABLE IF EXISTS {_TABLE}")
    await pool.execute(f"""
        CREATE TABLE {_TABLE} (
            id SERIAL PRIMARY KEY,
            parent_id INTEGER REFERENCES {_TABLE}(id),
            content TEXT NOT NULL,
            embedding vector(3),
            metadata JSONB
        )
    """)


async def _seed_data(pool):
    """Insert test rows."""
    await pool.execute(f"DELETE FROM {_TABLE}")
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


class TestPgvectorRetrieve:
    @pytest.fixture
    async def retriever(self):
        """Create a pgvector retriever connected to a real database.

        Tries Docker, then local PostgreSQL.  Skips if none are available.
        """
        pool, reason = await _try_connect()
        if pool is None:
            pytest.skip(f"PostgreSQL not available: {reason}")

        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        await _ensure_schema(pool)
        r = ConfigurablePgvectorRetriever(
            table=_TABLE,
            id_column="id",
            content_column="content",
            embedding_column="embedding",
            metadata_column="metadata",
            distance_metric="cosine",
        )
        r._pool = pool
        r._initialised = True

        await _seed_data(pool)
        yield r
        await pool.execute(f"DROP TABLE IF EXISTS {_TABLE}")
        await pool.close()

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


class TestPgvectorIdentifierValidation:
    def test_invalid_table_name_rejected(self):
        """Table names with semicolons etc. raise ConfigurationError."""
        from raglan.exceptions import ConfigurationError
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        with pytest.raises(ConfigurationError, match="unsafe characters"):
            ConfigurablePgvectorRetriever(table="kb_chunks; DROP TABLE kb_chunks;--")

    def test_invalid_column_name_rejected(self):
        from raglan.exceptions import ConfigurationError
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        with pytest.raises(ConfigurationError, match="unsafe characters"):
            ConfigurablePgvectorRetriever(
                table="kb_chunks",
                id_column="id' OR '1'='1",
            )

    def test_valid_identifiers_accepted(self):
        from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever

        # Schema-qualified with dots should pass
        r = ConfigurablePgvectorRetriever(
            table="public.kb_chunks",
            id_column="id",
            content_column="content",
            embedding_column="embedding",
        )
        assert r._table == "public.kb_chunks"
