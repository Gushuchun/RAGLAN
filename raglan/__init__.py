"""Raglan — a lightweight, highly configurable RAG retrieval engine."""

from raglan.exceptions import (
    CircuitBreakerOpenError,
    ConfigurationError,
    ContextBuilderError,
    EmbedderError,
    ExpanderError,
    FilterError,
    RaglanError,
    RerankerError,
    RetrieverError,
    StageError,
    TimeoutError,
)
from raglan.observability import LoggingMetricsCollector, NoOpMetricsCollector
from raglan.pipeline import Pipeline
from raglan.raglan import Raglan, RaglanBuilder
from raglan.types import (
    Filter,
    Op,
    PipelineContext,
    ScoredChunk,
    SearchOptions,
    SearchResult,
    StageDegradation,
    StageTiming,
    Trace,
    TraceLevel,
)

__all__ = [
    "CircuitBreakerOpenError",
    "ConfigurationError",
    "ContextBuilderError",
    "EmbedderError",
    "ExpanderError",
    "Filter",
    "FilterError",
    "LoggingMetricsCollector",
    "NoOpMetricsCollector",
    "Op",
    "Pipeline",
    "PipelineContext",
    "Raglan",
    "RaglanBuilder",
    "RaglanError",
    "RerankerError",
    "RetrieverError",
    "ScoredChunk",
    "SearchOptions",
    "SearchResult",
    "StageDegradation",
    "StageError",
    "StageTiming",
    "TimeoutError",
    "Trace",
    "TraceLevel",
]
