"""Circuit-breaker middleware — skips a stage after repeated failures."""

from __future__ import annotations

import asyncio
import time as _time
from typing import Any

from raglan.types import PipelineContext, StageDegradation


class CircuitBreakerMiddleware:
    """State machine that opens after *failure_threshold* consecutive
    failures, stays open for *recovery_timeout* seconds, then enters
    half-open to test the water.

    While open the stage is skipped entirely — no call is made.

    All state transitions are protected by an internal ``asyncio.Lock``
    so the middleware is safe for concurrent use across multiple requests.
    """

    name = "circuit_breaker"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._failures = 0
        self._state: str = "closed"
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    async def wrap(
        self,
        ctx: PipelineContext,
        next_stage: Any,
    ) -> PipelineContext:
        stage_name = getattr(next_stage, "__name__", "unknown")

        # --- state inspection & transition (locked) ---
        async with self._lock:
            if self._state == "open":
                if _time.monotonic() - self._opened_at >= self._recovery:
                    self._state = "half_open"
                else:
                    ctx.degradations.append(
                        StageDegradation(stage=stage_name, error="circuit breaker open")
                    )
                    return ctx

        # --- execute the stage (lock released) ---
        try:
            result = await next_stage(ctx)
        except Exception:
            async with self._lock:
                self._failures += 1
                if self._failures >= self._threshold:
                    self._state = "open"
                    self._opened_at = _time.monotonic()
            raise

        # --- success path (locked) ---
        async with self._lock:
            self._state = "closed"
            self._failures = 0
        return result  # type: ignore[no-any-return]
