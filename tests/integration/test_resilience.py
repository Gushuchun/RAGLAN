"""Tests for the resilience module — retry budget, rate limiter, health checker."""

from __future__ import annotations

import asyncio

import pytest

from raglan.resilience import HealthChecker, HealthStatus, RateLimiter, RetryBudget
from raglan.types import PipelineContext


class TestRetryBudget:
    async def test_acquire_within_budget(self):
        budget = RetryBudget(max_retries_per_window=3, window_seconds=60.0)
        assert await budget.acquire() is True
        assert await budget.acquire() is True
        assert await budget.acquire() is True

    async def test_acquire_exhausted(self):
        budget = RetryBudget(max_retries_per_window=2, window_seconds=60.0)
        assert await budget.acquire() is True
        assert await budget.acquire() is True
        assert await budget.acquire() is False

    async def test_available_counts_correctly(self):
        budget = RetryBudget(max_retries_per_window=5, window_seconds=60.0)
        assert budget.available == 5
        await budget.acquire()
        assert budget.available == 4

    async def test_window_expires(self):
        budget = RetryBudget(max_retries_per_window=2, window_seconds=0.01)
        await budget.acquire()
        await budget.acquire()
        assert await budget.acquire() is False
        await asyncio.sleep(0.02)
        assert await budget.acquire() is True

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="max_retries_per_window"):
            RetryBudget(max_retries_per_window=0)
        with pytest.raises(ValueError, match="window_seconds"):
            RetryBudget(max_retries_per_window=5, window_seconds=0)

    async def test_concurrent_access(self):
        budget = RetryBudget(max_retries_per_window=50, window_seconds=60.0)

        async def worker():
            results = []
            for _ in range(5):
                results.append(await budget.acquire())
            return results

        tasks = [asyncio.create_task(worker()) for _ in range(10)]
        all_results = await asyncio.gather(*tasks)
        acquired = sum(sum(r) for r in all_results)
        assert acquired == 50
        assert budget.available == 0


class TestRateLimiter:
    def test_invalid_rate(self):
        with pytest.raises(ValueError, match="rate must be"):
            RateLimiter(rate=0)
        with pytest.raises(ValueError, match="rate must be"):
            RateLimiter(rate=-1)

    async def test_acquire_within_limit(self):
        limiter = RateLimiter(rate=100, burst=10)
        for _ in range(10):
            assert await limiter.acquire() is True

    async def test_acquire_exhausted_burst(self):
        limiter = RateLimiter(rate=100, burst=3)
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False

    async def test_wait_and_acquire_times_out(self):
        limiter = RateLimiter(rate=0.1, burst=0)
        result = await limiter.wait_and_acquire(timeout=0.05)
        assert result is False

    async def test_wait_and_acquire_succeeds(self):
        limiter = RateLimiter(rate=1000, burst=5)
        result = await limiter.wait_and_acquire(timeout=1.0)
        assert result is True

    async def test_refill_over_time(self):
        limiter = RateLimiter(rate=50, burst=1)
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False
        await asyncio.sleep(0.05)
        assert await limiter.acquire() is True

    async def test_middleware_skips_on_exhausted(self):
        limiter = RateLimiter(rate=1, burst=1)
        # Drain the token
        assert await limiter.acquire() is True
        ctx = PipelineContext(query="test")

        async def _stage(c):
            c.final_results = ["result"]
            return c

        result = await limiter.wrap(ctx, _stage)
        assert len(result.degradations) == 1
        assert "rate limiter" in result.degradations[0].error

    async def test_middleware_passes_when_available(self):
        limiter = RateLimiter(rate=100, burst=10)
        ctx = PipelineContext(query="test")

        async def _stage(c):
            c.final_results = ["ok"]
            return c

        result = await limiter.wrap(ctx, _stage)
        assert result.final_results == ["ok"]


class TestHealthChecker:
    async def test_empty_checks(self):
        checker = HealthChecker()
        statuses = await checker.check_all()
        assert statuses == {}

    async def test_healthy_check(self):
        checker = HealthChecker({"db": lambda: asyncio.sleep(0, result=True)})

        # Async lambda can't return; use a proper async function
        async def _ok():
            return True

        checker = HealthChecker({"db": _ok})
        statuses = await checker.check_all()
        assert statuses["db"].healthy is True
        assert statuses["db"].message == "OK"
        assert statuses["db"].latency_ms >= 0

    async def test_unhealthy_check(self):
        async def _fail():
            return False

        checker = HealthChecker({"api": _fail})
        statuses = await checker.check_all()
        assert statuses["api"].healthy is False
        assert statuses["api"].message == "unhealthy"

    async def test_check_timeout(self):
        async def _slow():
            await asyncio.sleep(1.0)
            return True

        checker = HealthChecker({"slow": _slow})
        statuses = await checker.check_all(timeout=0.05)
        assert statuses["slow"].healthy is False
        assert "timeout" in statuses["slow"].message.lower()

    async def test_check_exception(self):
        async def _error():
            raise ConnectionError("refused")

        checker = HealthChecker({"broken": _error})
        statuses = await checker.check_all()
        assert statuses["broken"].healthy is False
        assert "refused" in statuses["broken"].message

    async def test_register_unregister(self):
        checker = HealthChecker()
        assert checker.check_names == []

        async def _ok():
            return True

        checker.register("test", _ok)
        assert "test" in checker.check_names
        assert len(checker.check_names) == 1

        checker.unregister("test")
        assert checker.check_names == []
        checker.unregister("nonexistent")  # no-op

    def test_health_status_dataclass(self):
        status = HealthStatus(name="x", healthy=True, message="OK")
        assert status.name == "x"
        assert status.healthy is True
        assert status.message == "OK"
        assert status.latency_ms == 0.0
        assert status.checked_at > 0

    async def test_parallel_execution(self):
        started: list[str] = []

        async def _check_a():
            started.append("a")
            await asyncio.sleep(0.02)
            return True

        async def _check_b():
            started.append("b")
            await asyncio.sleep(0.02)
            return True

        checker = HealthChecker({"a": _check_a, "b": _check_b})
        statuses = await checker.check_all(timeout=1.0)
        assert statuses["a"].healthy is True
        assert statuses["b"].healthy is True
        assert set(started) == {"a", "b"}  # both started
