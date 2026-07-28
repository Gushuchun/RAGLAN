"""Timeout middleware — fails a stage if it exceeds a deadline."""

from __future__ import annotations

import asyncio
from typing import Any

from raglan.types import PipelineContext, StageDegradation


class TimeoutMiddleware:
    """Caps a stage's execution time.  On timeout the stage is skipped
    and a ``StageDegradation`` is appended to the context."""

    name = "timeout"

    def __init__(self, timeout: float) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._timeout = timeout

    async def wrap(
        self,
        ctx: PipelineContext,
        next_stage: Any,
    ) -> PipelineContext:
        try:
            return await asyncio.wait_for(next_stage(ctx), timeout=self._timeout)
        except asyncio.TimeoutError:
            stage_name = getattr(next_stage, "__name__", "unknown")
            ctx.degradations.append(
                StageDegradation(
                    stage=stage_name,
                    error=f"timed out after {self._timeout:.1f}s",
                )
            )
            return ctx
