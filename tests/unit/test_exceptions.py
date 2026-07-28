"""Tests covering the exception hierarchy."""

from __future__ import annotations

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


def test_exception_inheritance():
    """All exceptions inherit from RaglanError."""
    assert issubclass(ConfigurationError, RaglanError)
    assert issubclass(StageError, RaglanError)
    assert issubclass(RetrieverError, StageError)
    assert issubclass(EmbedderError, StageError)
    assert issubclass(ExpanderError, StageError)
    assert issubclass(RerankerError, StageError)
    assert issubclass(ContextBuilderError, StageError)
    assert issubclass(TimeoutError, RaglanError)
    assert issubclass(CircuitBreakerOpenError, RaglanError)
    assert issubclass(FilterError, RaglanError)


def test_stage_error_formatting():
    e = StageError("retriever", "connection refused")
    assert "[retriever]" in str(e)
    assert "connection refused" in str(e)


def test_retriever_error_carries_name():
    e = RetrieverError("bm25", "corpus empty")
    assert e.retriever_name == "bm25"
    assert "[retriever]" in str(e)
    assert "bm25" in str(e)


def test_configuration_error():
    e = ConfigurationError("no retrievers configured")
    assert isinstance(e, RaglanError)


def test_timeout_error():
    e = TimeoutError("expander", 5.0)
    assert e.stage == "expander"
    assert e.timeout_s == 5.0
    assert "5.0s" in str(e)


def test_circuit_breaker_open_error():
    e = CircuitBreakerOpenError("reranker")
    assert e.stage == "reranker"
    assert "circuit breaker" in str(e).lower()


def test_filter_error():
    e = FilterError("unsupported operator: XOR")
    assert isinstance(e, RaglanError)


def test_base_exception_can_be_caught():
    """User code can catch RaglanError to handle all library errors."""
    errors = [
        ConfigurationError("a"),
        StageError("b"),
        TimeoutError("c", 1.0),
        FilterError("d"),
    ]
    count = 0
    for exc in errors:
        try:
            raise exc
        except RaglanError:
            count += 1
    assert count == 4
