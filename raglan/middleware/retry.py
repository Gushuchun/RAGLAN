"""Retry middleware — transparently retries a stage on transient failures."""

from __future__ import annotations

import asyncio
from typing import Any

from raglan.types import PipelineContext, StageDegradation


class RetryMiddleware:
    """Retries a stage up to *max_retries* times with configurable backoff.

    Only exceptions listed in *retryable* trigger a retry; all others are
    re-raised immediately.  After exhausting retries a ``StageDegradation``
    is recorded.
    """

    name = "retry"

    def __init__(
        self,
        max_retries: int = 3,
        backoff: str = "exponential",
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        retryable: tuple[type[BaseException], ...] = (
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ) -> None:
        if backoff not in ("exponential", "linear", "constant"):
            raise ValueError(
                f"backoff must be 'exponential', 'linear', or 'constant', got '{backoff}'"
            )
        self._max_retries = max_retries
        self._backoff = backoff
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._retryable = retryable

    async def wrap(
        self,
        ctx: PipelineContext,
        next_stage: Any,
    ) -> PipelineContext:
        stage_name = getattr(next_stage, "__name__", "unknown")
        last_error: BaseException | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await next_stage(ctx)  # type: ignore[no-any-return]
            except self._retryable as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = self._calc_delay(attempt)
                    await asyncio.sleep(delay)
            except Exception:
                raise

        ctx.degradations.append(
            StageDegradation(
                stage=stage_name,
                error=(f"failed after {self._max_retries + 1} attempts: {last_error}"),
            )
        )
        return ctx

    def _calc_delay(self, attempt: int) -> float:
        if self._backoff == "exponential":
            return float(min(self._initial_delay * (2**attempt), self._max_delay))
        elif self._backoff == "linear":
            return float(min(self._initial_delay * (attempt + 1), self._max_delay))
        return float(self._initial_delay)
