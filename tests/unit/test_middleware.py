"""Tests for middleware: Timeout, Retry, CircuitBreaker, Logging."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from raglan.middleware.circuit_breaker import CircuitBreakerMiddleware
from raglan.middleware.logging import LoggingMiddleware
from raglan.middleware.retry import RetryMiddleware
from raglan.middleware.timeout import TimeoutMiddleware
from raglan.types import PipelineContext

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ctx() -> PipelineContext:
    return PipelineContext(query="test")


class _SuccessStage:
    name = "test_stage"

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        ctx.metadata["called"] = True
        return ctx


class _FailingStage:
    name = "failing_stage"

    def __init__(self, exc: BaseException | None = None, delay: float = 0.0) -> None:
        self.exc = exc if exc is not None else RuntimeError("fail")
        self.delay = delay
        self.call_count = 0

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        self.call_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        raise self.exc


# ---------------------------------------------------------------------------
# TimeoutMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_passes_through_on_success():
    mw = TimeoutMiddleware(5.0)
    ctx = await mw.wrap(_ctx(), _SuccessStage())
    assert ctx.metadata.get("called") is True
    assert not ctx.degradations


@pytest.mark.asyncio
async def test_timeout_degradation_on_timeout():
    mw = TimeoutMiddleware(0.01)
    stage = _FailingStage(delay=1.0)  # will be timed out, not fail
    ctx = await mw.wrap(_ctx(), stage)
    assert len(ctx.degradations) == 1
    assert "timed out" in ctx.degradations[0].error


@pytest.mark.asyncio
async def test_timeout_propagates_real_error():
    mw = TimeoutMiddleware(5.0)
    stage = _FailingStage(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await mw.wrap(_ctx(), stage)


def test_timeout_rejects_zero():
    with pytest.raises(ValueError, match="> 0"):
        TimeoutMiddleware(0)


def test_timeout_rejects_negative():
    with pytest.raises(ValueError, match="> 0"):
        TimeoutMiddleware(-1.0)


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_passes_through_on_success():
    mw = RetryMiddleware(max_retries=2)
    ctx = await mw.wrap(_ctx(), _SuccessStage())
    assert ctx.metadata.get("called") is True


@pytest.mark.asyncio
async def test_retry_on_transient_error():
    mw = RetryMiddleware(max_retries=2, initial_delay=0.0)
    failing = _FailingStage(ConnectionError("transient"))
    ctx = await mw.wrap(_ctx(), failing)
    assert failing.call_count == 3  # 1 initial + 2 retries
    assert any("failed after 3 attempts" in d.error for d in ctx.degradations)


@pytest.mark.asyncio
async def test_retry_non_retryable_raises_immediately():
    mw = RetryMiddleware(
        max_retries=3,
        initial_delay=0.0,
        retryable=(ConnectionError,),  # RuntimeError is NOT retryable
    )
    failing = _FailingStage(RuntimeError("fatal"))
    with pytest.raises(RuntimeError, match="fatal"):
        await mw.wrap(_ctx(), failing)
    assert failing.call_count == 1  # no retries


# ---------------------------------------------------------------------------
# CircuitBreakerMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_circuit_breaker_closed_by_default():
    cb = CircuitBreakerMiddleware(failure_threshold=3)
    assert cb._state == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreakerMiddleware(failure_threshold=2, recovery_timeout=999.0)
    for _ in range(2):
        with contextlib.suppress(RuntimeError):
            await cb.wrap(_ctx(), _FailingStage(RuntimeError("fail")))
    assert cb._state == "open"


@pytest.mark.asyncio
async def test_circuit_breaker_records_degradation_when_open():
    cb = CircuitBreakerMiddleware(failure_threshold=1, recovery_timeout=999.0)
    with contextlib.suppress(RuntimeError):
        await cb.wrap(_ctx(), _FailingStage(RuntimeError("fail")))
    ctx = await cb.wrap(_ctx(), _SuccessStage())  # should be skipped
    assert any("circuit breaker open" in d.error for d in ctx.degradations)


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logging_records_success():
    logger = logging.getLogger("test_raglan_logging")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())  # avoid propagating
    mw = LoggingMiddleware(logger=logger, level=logging.DEBUG)
    ctx = await mw.wrap(_ctx(), _SuccessStage())
    assert ctx.metadata.get("called") is True
    logger.setLevel(logging.WARNING)  # clean up
