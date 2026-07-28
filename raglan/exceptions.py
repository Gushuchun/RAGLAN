"""Raglan exception hierarchy.

Every exception raised by Raglan inherits from ``RaglanError`` so that
callers can catch a single base type if they only care that *something*
went wrong inside the library.  Specific subclasses let advanced users
handle different failure modes differently.
"""

from __future__ import annotations

__all__ = [
    "CircuitBreakerOpenError",
    "ConfigurationError",
    "ContextBuilderError",
    "EmbedderError",
    "ExpanderError",
    "FilterError",
    "RaglanError",
    "RerankerError",
    "RetrieverError",
    "StageError",
    "TimeoutError",
]


class RaglanError(Exception):
    """Base class for all Raglan exceptions."""


# ============================================================================
# Configuration errors — raised before search begins
# ============================================================================


class ConfigurationError(RaglanError):
    """The pipeline configuration is invalid (e.g. no retrievers configured)."""


# ============================================================================
# Stage-level errors
# ============================================================================


class StageError(RaglanError):
    """A pipeline stage failed and fallback is disabled."""

    def __init__(self, stage: str, message: str = "") -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}" if message else stage)


class RetrieverError(StageError):
    """A retriever failed."""

    def __init__(self, retriever_name: str, message: str = "") -> None:
        super().__init__(stage="retriever", message=f"{retriever_name}: {message}")
        self.retriever_name = retriever_name


class EmbedderError(StageError):
    """Embedding generation failed."""


class ExpanderError(StageError):
    """Query expansion (LLM call) failed."""


class RerankerError(StageError):
    """Re-ranking failed."""


class ContextBuilderError(StageError):
    """Context builder (e.g. parent-chunk loading) failed."""


# ============================================================================
# Resilience errors
# ============================================================================


class TimeoutError(RaglanError):
    """A pipeline stage timed out."""

    def __init__(self, stage: str, timeout_s: float) -> None:
        self.stage = stage
        self.timeout_s = timeout_s
        super().__init__(f"[{stage}] timed out after {timeout_s:.1f}s")


class CircuitBreakerOpenError(RaglanError):
    """A stage was skipped because its circuit breaker is open."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] circuit breaker is open")


# ============================================================================
# Filter errors
# ============================================================================


class FilterError(RaglanError):
    """A filter expression is invalid for the target retriever."""


# ============================================================================
# Convenience — re-exported at package level
# ============================================================================

__all__ = [
    "CircuitBreakerOpenError",
    "ConfigurationError",
    "ContextBuilderError",
    "EmbedderError",
    "ExpanderError",
    "FilterError",
    "RaglanError",
    "RerankerError",
    "RetrieverError",
    "StageError",
    "TimeoutError",
]
