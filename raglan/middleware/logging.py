"""Structured-logging middleware for pipeline observability."""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from raglan.types import PipelineContext

_logger = logging.getLogger("raglan.middleware.logging")


class LoggingMiddleware:
    """Logs stage entry, exit, timing, and any degradation.

    Parameters
    ----------
    logger:
        A ``logging.Logger`` instance.  If ``None``, the module-level
        ``raglan.middleware.logging`` logger is used.
    level:
        Log level for normal stage completion messages.
    """

    name = "logging"

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
    ) -> None:
        self._logger = logger or _logger
        self._level = level

    async def wrap(
        self,
        ctx: PipelineContext,
        next_stage: Any,
    ) -> PipelineContext:
        stage_name = getattr(next_stage, "__name__", "unknown")
        t0 = _time.monotonic()
        self._logger.log(self._level, "[%s] started", stage_name)
        try:
            ctx = await next_stage(ctx)
        except Exception as exc:
            self._logger.error("[%s] failed: %s", stage_name, exc)
            raise
        else:
            elapsed = (_time.monotonic() - t0) * 1000
            self._logger.log(
                self._level,
                "[%s] completed in %.1fms",
                stage_name,
                elapsed,
            )
        return ctx
