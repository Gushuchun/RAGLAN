"""Health checker — async health checks for pipeline dependencies."""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class HealthStatus:
    """Result of a health check."""

    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    checked_at: float = field(default_factory=_time.monotonic)


HealthCheck = Callable[[], Awaitable[bool]]


class HealthChecker:
    """Runs async health checks against pipeline dependencies.

    Parameters
    ----------
    checks:
        Mapping of ``name -> async callable``.  Each callable should
        return ``True`` if the dependency is healthy.
    """

    def __init__(self, checks: dict[str, HealthCheck] | None = None) -> None:
        self._checks: dict[str, HealthCheck] = dict(checks or {})

    def register(self, name: str, check: HealthCheck) -> None:
        """Add a named health check."""
        self._checks[name] = check

    def unregister(self, name: str) -> None:
        """Remove a named health check."""
        self._checks.pop(name, None)

    async def check_all(self, timeout: float = 5.0) -> dict[str, HealthStatus]:
        """Run all registered health checks in parallel.

        Parameters
        ----------
        timeout:
            Per-check timeout in seconds.
        """
        if not self._checks:
            return {}

        async def _run_one(name: str, check: HealthCheck) -> HealthStatus:
            t0 = _time.monotonic()
            try:
                ok = await asyncio.wait_for(check(), timeout=timeout)
                return HealthStatus(
                    name=name,
                    healthy=ok,
                    message="OK" if ok else "unhealthy",
                    latency_ms=(_time.monotonic() - t0) * 1000,
                )
            except asyncio.TimeoutError:
                return HealthStatus(
                    name=name,
                    healthy=False,
                    message=f"timeout after {timeout:.1f}s",
                    latency_ms=timeout * 1000,
                )
            except Exception as exc:
                return HealthStatus(
                    name=name,
                    healthy=False,
                    message=str(exc),
                    latency_ms=(_time.monotonic() - t0) * 1000,
                )

        tasks = {
            name: asyncio.create_task(_run_one(name, check)) for name, check in self._checks.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=False))

    @property
    def check_names(self) -> list[str]:
        """Names of registered health checks."""
        return list(self._checks)
