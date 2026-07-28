"""Core data types for Raglan.

Every structured value that flows through the pipeline is defined here.
These types have zero external dependencies — they are plain Python
dataclasses and enums that can be imported anywhere.
"""

from __future__ import annotations

import time as _time

__all__ = [
    "Filter",
    "Op",
    "PipelineContext",
    "ScoredChunk",
    "SearchOptions",
    "SearchResult",
    "StageDegradation",
    "StageTiming",
    "Trace",
    "TraceLevel",
]
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Chunk & result types
# ============================================================================


@dataclass
class ScoredChunk:
    """A chunk returned by a Retriever, carrying a relevance score.

    This is the universal currency that flows between Stage 2 (Retrieve),
    Stage 3 (Fusion), and Stage 4 (Rerank).
    """

    chunk_id: str
    content: str
    score: float
    parent_chunk_id: str | None = None
    chunk_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # e.g. "bm25", "pgvector"


@dataclass
class SearchResult:
    """Final result returned to the caller after all pipeline stages.

    Differs from ``ScoredChunk`` in that ``parent_content`` may carry
    expanded context loaded by the ContextBuilder stage.
    """

    chunk_id: str
    content: str
    score: float
    parent_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # e.g. "bm25", "pgvector→cross_encoder"


# ============================================================================
# Filter system
# ============================================================================


class Op(str, Enum):
    """Filter comparison operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    EXISTS = "exists"
    CONTAINS = "contains"
    AND = "and"
    OR = "or"


@dataclass
class Filter:
    """A logical filter tree for metadata-constrained search.

    Leaf nodes represent comparisons (``field`` is set, ``children`` is
    ``None``).  Internal nodes represent boolean combinations (``field``
    is ``None``, ``children`` is populated).

    Use the factory methods (``eq``, ``gte``, ``all``, ``any``, etc.)
    or the ``&`` / ``|`` operator overloads rather than constructing
    instances directly.
    """

    field: str | None = None
    op: Op = Op.EQ
    value: Any = None
    children: list[Filter] | None = None

    # -- factories -----------------------------------------------------------

    @staticmethod
    def eq(field: str, value: Any) -> Filter:
        """``field == value``."""
        return Filter(field=field, op=Op.EQ, value=value)

    @staticmethod
    def ne(field: str, value: Any) -> Filter:
        """``field != value``."""
        return Filter(field=field, op=Op.NE, value=value)

    @staticmethod
    def gt(field: str, value: Any) -> Filter:
        """``field > value``."""
        return Filter(field=field, op=Op.GT, value=value)

    @staticmethod
    def gte(field: str, value: Any) -> Filter:
        """``field >= value``."""
        return Filter(field=field, op=Op.GTE, value=value)

    @staticmethod
    def lt(field: str, value: Any) -> Filter:
        """``field < value``."""
        return Filter(field=field, op=Op.LT, value=value)

    @staticmethod
    def lte(field: str, value: Any) -> Filter:
        """``field <= value``."""
        return Filter(field=field, op=Op.LTE, value=value)

    @staticmethod
    def in_(field: str, values: list[Any]) -> Filter:
        """``field IN (values...)``."""
        return Filter(field=field, op=Op.IN, value=values)

    @staticmethod
    def exists(field: str) -> Filter:
        """``field IS NOT NULL`` (metadata key exists)."""
        return Filter(field=field, op=Op.EXISTS)

    @staticmethod
    def contains(field: str, value: str) -> Filter:
        """``field`` contains substring ``value``."""
        return Filter(field=field, op=Op.CONTAINS, value=value)

    @staticmethod
    def all(*filters: Filter) -> Filter:
        """Logical AND of sub-filters."""
        return Filter(op=Op.AND, children=list(filters))

    @staticmethod
    def any(*filters: Filter) -> Filter:
        """Logical OR of sub-filters."""
        return Filter(op=Op.OR, children=list(filters))

    # -- operator overloads --------------------------------------------------

    def __and__(self, other: Filter) -> Filter:
        return Filter.all(self, other)

    def __or__(self, other: Filter) -> Filter:
        return Filter.any(self, other)

    def __bool__(self) -> bool:
        # Prevent accidental truthiness checks on a Filter object.
        raise TypeError(
            "Filter objects cannot be evaluated as booleans. "
            "Use `filter.field` or `filter.children` explicitly."
        )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SearchOptions:
    """Per-request overrides for pipeline behaviour.

    Any field left at its default (``None`` / ``-1``) means "use the
    pipeline-level default".
    """

    top_k: int = -1
    dense_top_k: int = -1
    bm25_top_k: int = -1
    reranker_min_score: float = -1.0
    reranker_top_n: int = -1
    max_context_tokens: int = -1
    retriever_timeout: float | None = None
    fallback_mode: str = ""


# ============================================================================
# Trace & observability
# ============================================================================


class TraceLevel(str, Enum):
    """Granularity of pipeline trace collection.

    Reserved for future use — the pipeline engine always produces full
    trace data.  A future release will add filtering based on this level.
    """

    MINIMAL = "minimal"  # timings + counts only, no content
    NORMAL = "normal"  # + degradation records, per-stage metadata
    FULL = "full"  # + raw intermediate results (debug only)


@dataclass
class StageDegradation:
    """Record of a stage that was skipped due to an error."""

    stage: str
    error: str
    retriever: str = ""  # populated for retrieval-stage degradations


@dataclass
class StageTiming:
    """Wall-clock timing for a single pipeline stage."""

    stage: str
    elapsed_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Post-search trace produced by the pipeline.

    Content detail depends on the configured ``TraceLevel``.
    """

    query: str
    total_ms: float = 0.0
    stage_timings: list[StageTiming] = field(default_factory=list)
    degradations: list[StageDegradation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- convenience ---------------------------------------------------------

    @property
    def degraded(self) -> bool:
        """True when at least one stage was skipped."""
        return len(self.degradations) > 0

    @property
    def degraded_stage_names(self) -> list[str]:
        """Names of stages that were degraded."""
        return [d.stage for d in self.degradations]


# ============================================================================
# Pipeline context — mutable state bag that flows through the pipeline
# ============================================================================


@dataclass
class PipelineContext:
    """Mutable context object passed from stage to stage.

    Each stage reads its inputs from this object and writes its outputs
    back to it.  The pipeline engine is responsible for creating it and
    extracting the final results from it.
    """

    # -- user input (set once) ------------------------------------------------
    query: str = ""
    filters: list[Filter] | None = None
    options: SearchOptions = field(default_factory=SearchOptions)

    # -- Stage 1: QueryExpander -----------------------------------------------
    expanded_queries: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)

    # -- Stage 1→2 bridge: Embedder ------------------------------------------
    embeddings: list[list[float]] = field(default_factory=list)

    # -- Stage 2: Retriever(s) ------------------------------------------------
    retriever_results: dict[str, list[list[ScoredChunk]]] = field(default_factory=dict)

    # -- Stage 3: Fusion ------------------------------------------------------
    fused_candidates: list[ScoredChunk] = field(default_factory=list)

    # -- Stage 4: Reranker ----------------------------------------------------
    reranked_candidates: list[ScoredChunk] = field(default_factory=list)

    # -- Stage 5: ContextBuilder ----------------------------------------------
    final_results: list[SearchResult] = field(default_factory=list)

    # -- metadata -------------------------------------------------------------
    started_at: float = field(default_factory=_time.monotonic)
    degradations: list[StageDegradation] = field(default_factory=list)
    stage_timings: list[StageTiming] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
