"""Rate-limiter middleware — token-bucket throttling for pipeline stages."""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

from raglan.types import PipelineContext, StageDegradation


class RateLimiter:
    """Token-bucket rate limiter that can be used as pipeline middleware.

    Parameters
    ----------
    rate:
        Maximum number of requests per second.
    burst:
        Maximum burst size (bucket capacity).  Defaults to *rate*.
    """

    name = "rate_limiter"

    def __init__(self, rate: float, burst: float | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._rate = rate
        self._burst = burst if burst is not None else rate
        self._tokens = self._burst
        self._last_refill = _time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire one token.

        Returns ``True`` if a token was acquired, ``False`` if none are
        available and the caller should wait or back off.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    async def wait_and_acquire(self, timeout: float | None = None) -> bool:
        """Wait until a token becomes available or *timeout* is exceeded."""
        deadline = _time.monotonic() + timeout if timeout is not None else None
        while True:
            if await self.acquire():
                return True
            if deadline is not None and _time.monotonic() >= deadline:
                return False
            wait = max(0.01, 1.0 / self._rate)
            await asyncio.sleep(min(wait, 0.1))

    def _refill(self) -> None:
        now = _time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def wrap(
        self,
        ctx: PipelineContext,
        next_stage: Any,
    ) -> PipelineContext:
        """Middleware entry point — acquires a token before proceeding.

        If the rate limiter is exhausted, the stage is skipped and a
        ``StageDegradation`` is recorded.
        """
        if not await self.acquire():
            stage_name = getattr(next_stage, "__name__", "unknown")
            ctx.degradations.append(
                StageDegradation(stage=stage_name, error="rate limiter exhausted")
            )
            return ctx
        return await next_stage(ctx)  # type: ignore[no-any-return]
