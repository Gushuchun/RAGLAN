"""Observability — metrics collectors for pipeline monitoring.

Provides a ``MetricsCollector`` protocol-based interface and a default
no-op implementation.  Users can inject custom collectors to send metrics
to Prometheus, Datadog, OpenTelemetry, etc.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger("raglan.observability")


class NoOpMetricsCollector:
    """Default metrics collector that does nothing.

    Use this when no external observability system is configured.
    """

    name = "noop"

    async def record_search(
        self,
        query: str,
        total_ms: float,
        result_count: int,
        degraded: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    async def record_stage(
        self,
        stage_name: str,
        elapsed_ms: float,
        degraded: bool = False,
        error: str | None = None,
    ) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialise the collector for ``export_config()``."""
        return {"type": self.name, "params": {}}


class LoggingMetricsCollector:
    """Metrics collector that logs pipeline activity via Python logging.

    Useful for development and basic production monitoring without an
    external metrics infrastructure.

    Parameters
    ----------
    logger:
        Optional pre-configured logger.  When ``None``, a module-level
        logger named ``"raglan.observability"`` is used.
    level:
        Log level for successful operations.  Default ``logging.DEBUG``.
    log_queries:
        When ``True``, the raw query text is included in log messages.
        When ``False`` (the default), queries are redacted to prevent
        accidental PII/credential leakage in production logs.
    """

    name = "logging"

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.DEBUG,
        log_queries: bool = False,
    ) -> None:
        self._logger = logger or _logger
        self._level = level
        self._log_queries = log_queries

    async def record_search(
        self,
        query: str,
        total_ms: float,
        result_count: int,
        degraded: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        status = "degraded" if degraded else "ok"
        displayed_query = query[:200] if self._log_queries else "<redacted>"
        self._logger.log(
            self._level,
            "search completed: query=%r total_ms=%.1f results=%d status=%s",
            displayed_query,
            total_ms,
            result_count,
            status,
        )

    async def record_stage(
        self,
        stage_name: str,
        elapsed_ms: float,
        degraded: bool = False,
        error: str | None = None,
    ) -> None:
        if degraded:
            self._logger.warning(
                "stage degraded: %s %.1fms error=%s",
                stage_name,
                elapsed_ms,
                error,
            )
        else:
            self._logger.log(
                self._level,
                "stage completed: %s %.1fms",
                stage_name,
                elapsed_ms,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the collector for ``export_config()``."""
        return {"type": self.name, "params": {"log_queries": self._log_queries}}
