"""Configurable pgvector retriever — no-code adapter for Postgres + pgvector.

Point it at your table, tell it which columns hold what, and it
auto-generates the SQL.  The Protocol-based ``Retriever`` is still
available for users who need full control; this module covers the 80 %
case where the mapping from table columns to Raglan concepts is
straightforward.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, ClassVar

from raglan.exceptions import ConfigurationError, FilterError
from raglan.types import Filter, Op, ScoredChunk

logger = logging.getLogger(__name__)

# Only allow schema-qualified identifiers: letters, digits, underscore, dot
_SQL_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]*$")

# Injects extra parameterised WHERE predicates.
# (session_or_pool, request, base_param_count) -> (predicates_with_$N, extra_params) | None
_WhereBuilder = Callable[[Any, dict[str, Any] | None, int], tuple[list[str], list[Any]] | None]


def _validate_sql_identifier(name: str, label: str) -> None:
    """Raise ``ConfigurationError`` if *name* is not a safe SQL identifier."""
    if not _SQL_IDENTIFIER_RE.match(name):
        raise ConfigurationError(
            f"{label} '{name}' contains unsafe characters. "
            f"Only alphanumeric characters, underscores, and dots are allowed."
        )


def _to_sqlalchemy_params(sql: str, params: list[Any]) -> tuple[str, dict[str, Any]]:
    """Rewrite asyncpg positional params (``$1``, ``$2``, ...) as SQLAlchemy named params.

    Returns ``(sql, {"p1": ..., "p2": ...})`` suitable for
    ``session.execute(text(sql), named_params)``.

    PostgreSQL ``:pN::type`` casts are rewritten to ``CAST(:pN AS type)``
    because SQLAlchemy's ``text()`` mis-parses the ``::`` cast into a stray
    bound parameter.
    """
    named_sql = re.sub(r"\$(\d+)", r":p\1", sql)
    new_sql = re.sub(r":p(\d+)::(\w+)", r"CAST(:p\1 AS \2)", named_sql)
    named = {f"p{i + 1}": params[i] for i in range(len(params))}
    return new_sql, named


class ConfigurablePgvectorRetriever:
    """A pgvector-backed retriever configured via column-name mappings.

    Parameters
    ----------
    connection_string:
        ``postgresql://user:pass@host:port/db`` (asyncpg format).
    table:
        Fully-qualified table name (``"public.kb_chunks"``).
    id_column:
        Primary-key column.  Default ``"id"``.
    content_column:
        Column holding the chunk text.  Default ``"content"``.
    embedding_column:
        ``vector`` column.  Default ``"embedding"``.
    parent_id_column:
        Column referencing the parent chunk / document.  When ``None``
        every chunk is treated as its own parent.  Default ``None``.
    metadata_column:
        ``jsonb`` column for metadata filtering.  When ``None``
        metadata filters are silently ignored.  Default ``None``.
    distance_metric:
        ``"cosine"``, ``"l2"``, or ``"ip"`` (inner product).
        Default ``"cosine"``.
    connection_pool:
        Pre-built ``asyncpg.Pool``.  When supplied, *connection_string*
        is ignored.
    """

    name = "pgvector"
    requires_embeddings = True

    _DISTANCE_MAP: ClassVar[dict[str, str]] = {
        "cosine": "<=>",
        "l2": "<->",
        "ip": "<#>",
    }

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        table: str,
        id_column: str = "id",
        content_column: str = "content",
        embedding_column: str = "embedding",
        parent_id_column: str | None = None,
        metadata_column: str | None = None,
        distance_metric: str = "cosine",
        connection_pool: Any | None = None,
        session_factory: Any | None = None,
        where_builder: _WhereBuilder | None = None,
    ) -> None:
        if distance_metric not in self._DISTANCE_MAP:
            raise ValueError(
                f"distance_metric must be one of "
                f"{list(self._DISTANCE_MAP)}, got '{distance_metric}'"
            )

        # Validate SQL identifiers to prevent injection via configuration
        _validate_sql_identifier(table, "table")
        _validate_sql_identifier(id_column, "id_column")
        _validate_sql_identifier(content_column, "content_column")
        _validate_sql_identifier(embedding_column, "embedding_column")
        if parent_id_column is not None:
            _validate_sql_identifier(parent_id_column, "parent_id_column")
        if metadata_column is not None:
            _validate_sql_identifier(metadata_column, "metadata_column")

        self._conn_string = connection_string
        self._table = table
        self._id_col = id_column
        self._content_col = content_column
        self._embedding_col = embedding_column
        self._parent_id_col = parent_id_column
        self._metadata_col = metadata_column
        self._dist_op = self._DISTANCE_MAP[distance_metric]
        self._pool: Any = connection_pool
        self._pool_lock = asyncio.Lock()

        # SQLAlchemy async session factory — mutually exclusive with the
        # asyncpg connection string / pool.
        self._session_factory = session_factory
        self._using_sqlalchemy = session_factory is not None

        # Optional callback injecting extra parameterised WHERE predicates.
        # Signature: (session_or_pool, request, base_param_count) ->
        # (predicates_with_$N, extra_params) | None.
        self._where_builder = where_builder

        # Lazy-init flag
        self._initialised = False

    # ------------------------------------------------------------------
    # Retriever protocol
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
        request: dict[str, Any] | None = None,
    ) -> list[list[ScoredChunk]]:
        """Search each embedding vector and return the top *top_k* chunks.

        *request* is passed to the configured ``where_builder`` (if any) so
        it can inject permission-scoping predicates (e.g. visibility/ACL).
        """
        await self._ensure_pool()
        filter_clause, filter_params = self._build_filter(filters)

        # Optional where_builder injects extra parameterised predicates.
        extra_predicates: list[str] = []
        extra_params: list[Any] = []
        if self._where_builder is not None:
            base_count = 2 + len(filter_params)  # $1=vector, $2=top_k, then filters
            built = self._where_builder(self._pool, request, base_count)
            if built:
                extra_predicates, extra_params = built

        where_parts = [filter_clause, *extra_predicates]
        where_clause = " AND ".join(where_parts) if where_parts else "TRUE"

        results: list[list[ScoredChunk]] = []
        for emb in embeddings:
            vec_str = _format_vector(emb)
            parent_sel = f"{self._parent_id_col}" if self._parent_id_col else f"{self._id_col}"
            sql = (
                f"SELECT {self._id_col}, {self._content_col}, {parent_sel}, "  # nosec B608
                f"1 - ({self._embedding_col} {self._dist_op} $1::vector) AS score "
                f"FROM {self._table} "
                f"WHERE {where_clause} "
                f"ORDER BY {self._embedding_col} {self._dist_op} $1::vector "
                f"LIMIT $2"
            )
            params = [vec_str, top_k, *filter_params, *extra_params]

            rows = await self._fetch_all(sql, params, timeout=timeout)
            results.append(
                [
                    ScoredChunk(
                        chunk_id=str(r[0]),
                        content=str(r[1]),
                        score=float(r[3]),
                        parent_chunk_id=str(r[2]) if r[2] is not None else None,
                        source=self.name,
                    )
                    for r in rows
                ]
            )

        return results

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        """Not implemented — pgvector manages its own index."""
        # Users manage their pgvector data externally.
        # This retriever is read-only by design.

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Not implemented — use INSERT directly on your table."""

    async def remove(self, chunk_ids: list[str]) -> None:
        """Not implemented — use DELETE directly on your table."""

    # ------------------------------------------------------------------
    # Convenience: parent-chunk loader for ParentExpander
    # ------------------------------------------------------------------

    async def load_parents(self, chunk_ids: list[str]) -> dict[str, str]:
        """Load parent content for a list of chunk IDs.

        When *parent_id_column* is set, loads the full content of each
        chunk's parent document.  Returns ``{chunk_id: parent_content}``.
        """
        if not chunk_ids or not self._parent_id_col:
            return {}

        await self._ensure_pool()
        sql = (
            f"SELECT DISTINCT ON (p.{self._id_col}) "  # nosec B608
            f"c.{self._id_col} AS child_id, "
            f"p.{self._content_col} AS parent_content "
            f"FROM {self._table} c "
            f"JOIN {self._table} p "
            f"  ON c.{self._parent_id_col} = p.{self._id_col} "
            f"WHERE c.{self._id_col} = ANY($1::text[])"
        )

        rows = await self._fetch_all(sql, chunk_ids)
        return {str(r[0]): str(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the connection pool and release all resources.

        In SQLAlchemy mode the session factory is owned by the caller and is
        not closed here.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialised = False

    async def _ensure_pool(self) -> None:
        if self._using_sqlalchemy:
            self._initialised = True
            return

        if self._pool is not None:
            # asyncpg pools are loop-bound — recreate if the loop changed.
            # Only check actual asyncpg Pool objects (skip mocks/test objects).
            if type(self._pool).__name__ == "Pool":
                try:
                    pool_loop = self._pool._loop
                    current_loop = asyncio.get_running_loop()
                    if pool_loop is not current_loop:
                        await self._pool.close()
                        self._pool = None
                except (AttributeError, RuntimeError):
                    pass
            if self._pool is not None:
                return

        async with self._pool_lock:
            if self._pool is not None:  # double-check
                return  # type: ignore[unreachable]
            from raglan._lazy import _import_module

            _import_module("asyncpg", hint="pip install raglan-retrieval[pgvector]")
            import asyncpg

            self._pool = await asyncpg.create_pool(self._conn_string, min_size=1, max_size=10)
            self._initialised = True

    async def _fetch_all(
        self, sql: str, params: list[Any], timeout: float | None = None
    ) -> list[Any]:
        """Execute *sql* and return rows indexable by position (``row[0]``).

        Dispatches on the configured connection mode — asyncpg pool or
        SQLAlchemy async session.
        """
        if not self._using_sqlalchemy:
            return await self._pool.fetch(sql, *params, timeout=timeout)  # type: ignore[no-any-return]

        from raglan._lazy import _import_module

        _import_module("sqlalchemy", hint="pip install sqlalchemy")
        from sqlalchemy import text

        # Convert asyncpg positional params ($1, $2, ...) to SQLAlchemy named
        # params (:p1, :p2, ...).
        sqlalchemy_sql, named_params = _to_sqlalchemy_params(sql, params)
        assert self._session_factory is not None
        async with self._session_factory() as session:
            result = await session.execute(text(sqlalchemy_sql), named_params)
            return list(result.all())

    def _build_filter(
        self, filters: list[Filter] | None, base_param_count: int = 2
    ) -> tuple[str, list[Any]]:
        """Translate Raglan Filter tree to a SQL WHERE clause.

        When *metadata_column* is set, filters are applied against the
        JSONB column using the ``->>`` operator.  When it is ``None``,
        filters are silently ignored.

        Parameters
        ----------
        filters:
            The filter tree to translate.
        base_param_count:
            Number of parameters that precede the filter clause in the
            SQL statement (e.g. $1=vector, $2=top_k).
        """
        if not filters:
            return "TRUE", []

        if not self._metadata_col:
            logger.warning(
                "Filters were provided but metadata_column is not configured "
                "— filters will be ignored. Set metadata_column to enable "
                "metadata-based filtering."
            )
            return "TRUE", []

        params: list[Any] = []
        param_idx = base_param_count + 1

        def _walk(f: Filter) -> str:
            nonlocal param_idx

            if f.op in (Op.AND, Op.OR):
                if not f.children:
                    return "TRUE"
                joiner = " AND " if f.op == Op.AND else " OR "
                return "(" + joiner.join(_walk(c) for c in f.children) + ")"

            col_ref = f"{self._metadata_col}->>'{f.field}'"

            if f.op == Op.EQ:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"{col_ref} = ${idx}"
            elif f.op == Op.NE:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"{col_ref} != ${idx}"
            elif f.op == Op.GT:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"({col_ref})::numeric > ${idx}"
            elif f.op == Op.GTE:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"({col_ref})::numeric >= ${idx}"
            elif f.op == Op.LT:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"({col_ref})::numeric < ${idx}"
            elif f.op == Op.LTE:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"({col_ref})::numeric <= ${idx}"
            elif f.op == Op.IN:
                idx = param_idx
                param_idx += 1
                params.append(f.value)
                return f"{col_ref} = ANY(${idx}::text[])"
            elif f.op == Op.EXISTS:
                return f"{col_ref} IS NOT NULL"
            elif f.op == Op.CONTAINS:
                idx = param_idx
                param_idx += 1
                params.append(f"%{f.value}%")
                return f"{col_ref} LIKE ${idx}"
            else:
                raise FilterError(f"Unsupported filter operator: {f.op}")

        where = _walk(Filter.all(*filters))
        return where, params

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "connection_string": "<redacted>" if self._conn_string else None,
                "table": self._table,
                "id_column": self._id_col,
                "content_column": self._content_col,
                "embedding_column": self._embedding_col,
                "parent_id_column": self._parent_id_col,
                "metadata_column": self._metadata_col,
                "distance_metric": {v: k for k, v in self._DISTANCE_MAP.items()}.get(
                    self._dist_op, "cosine"
                ),
            },
        }


def _format_vector(emb: list[float]) -> str:
    """Format a float list as a pgvector-compatible string with controlled precision."""
    return "[" + ",".join(f"{x:.8g}" for x in emb) + "]"
