"""Raglan protocol definitions.

All user-implementable interfaces are defined here as Protocols.
The core engine depends only on these signatures, never on concrete
implementations. Each protocol carries a ``name`` attribute used for
logging, trace attribution, and per-stage timeout configuration.

See ``docs/architecture.md`` for the rationale behind each protocol's design.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from raglan.types import Filter, ScoredChunk, SearchResult

# ---------------------------------------------------------------------------
# Forward reference — PipelineContext is defined in pipeline.py but needs
# to be referenced here so Middleware and Stage can be typed properly.
# ---------------------------------------------------------------------------
_PipelineContext = object  # resolved at runtime by pipeline.py


# ============================================================================
# Stage protocols
# ============================================================================


@runtime_checkable
class QueryExpander(Protocol):
    """Expand a user query into multiple search variants (Stage 1).

    The first element of the returned list MUST be the original query;
    subsequent elements are LLM-generated variants that capture different
    phrasings or aspects of the same information need.
    """

    name: str

    async def expand(self, query: str, num_variants: int = 3) -> tuple[list[str], dict[str, Any]]:
        """Return ``(expanded_queries, entities)``."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Convert text batches into dense embedding vectors.

    Called between Stage 1 and Stage 2. The batch interface lets
    implementations make a single API call for all query variants
    rather than N separate calls.
    """

    name: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Search for relevant chunks given a batch of queries (Stage 2).

    Dense retrievers consume ``embeddings``; sparse retrievers ignore it.
    The return type is ``list[list[ScoredChunk]]`` — one inner list per
    input query — so the fusion stage can correlate results back to the
    query that produced them.

    This is the minimal read-only protocol.  For retrievers that support
    index management (add / remove / rebuild), implement the extended
    :class:`IndexableRetriever` protocol.
    """

    name: str
    requires_embeddings: bool

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
        request: dict[str, Any] | None = None,
    ) -> list[list[ScoredChunk]]: ...


@runtime_checkable
class IndexableRetriever(Retriever, Protocol):
    """A :class:`Retriever` that also supports index management.

    Implement ``index``, ``add``, and ``remove`` for incremental index
    updates.  Read-only backends only need to implement :class:`Retriever`.
    """

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        """Build / rebuild the full-text index from a stream of batches."""
        ...

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Incrementally add chunks."""
        ...

    async def remove(self, chunk_ids: list[str]) -> None:
        """Incrementally remove chunks."""
        ...


@runtime_checkable
class Fusion(Protocol):
    """Merge results from multiple retrievers into a single ranked list (Stage 3)."""

    name: str

    async def fuse(
        self,
        retriever_results: dict[str, list[list[ScoredChunk]]],
        original_query_idx: int = 0,
    ) -> list[ScoredChunk]:
        """Return deduplicated, fused candidates ordered by relevance."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Fine-grained relevance re-ranking of a candidate list (Stage 4)."""

    name: str

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Return re-ranked top-k candidates, discarding scores below *min_score*."""
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Expand or transform chunk content for the final result set (Stage 5)."""

    name: str

    async def build(
        self,
        query: str,
        candidates: list[ScoredChunk],
        max_tokens: int = 6000,
    ) -> list[SearchResult]:
        """Return final ``SearchResult`` objects ready for the caller."""
        ...


# ============================================================================
# Middleware protocol
# ============================================================================


@runtime_checkable
class Middleware(Protocol):
    """Cross-cutting wrapper around a pipeline stage.

    A middleware that catches an exception should append a
    ``StageDegradation`` to the context and return it rather than
    re-raising — unless ``fallback_mode`` is ``"strict"``.
    """

    name: str

    async def wrap(
        self,
        ctx: _PipelineContext,
        next_stage: Stage,
    ) -> _PipelineContext: ...


# ============================================================================
# Observability protocol
# ============================================================================


@runtime_checkable
class MetricsCollector(Protocol):
    """Collect pipeline-level metrics for external observability systems.

    Implementations can send data to Prometheus, Datadog, OpenTelemetry,
    CloudWatch, or any other metrics backend.
    """

    name: str

    async def record_search(
        self,
        query: str,
        total_ms: float,
        result_count: int,
        degraded: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Called once per :meth:`Pipeline.run` invocation."""
        ...

    async def record_stage(
        self,
        stage_name: str,
        elapsed_ms: float,
        degraded: bool = False,
        error: str | None = None,
    ) -> None:
        """Called after each pipeline stage completes or fails."""
        ...


# ============================================================================
# Convenience type aliases
# ============================================================================

Stage = Callable[[_PipelineContext], Awaitable[_PipelineContext]]

ParentChunkLoader = Callable[[list[str]], Awaitable[dict[str, str]]]
