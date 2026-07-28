"""Retry budget — limits retry attempts across a time window.

Prevents retry storms by capping the total number of retries within a
configurable sliding window.
"""

from __future__ import annotations

import asyncio
import time as _time


class RetryBudget:
    """Token-bucket-style retry budget with a sliding window.

    Parameters
    ----------
    max_retries_per_window:
        Maximum number of retries allowed within *window_seconds*.
    window_seconds:
        Duration of the sliding window in seconds.
    """

    def __init__(
        self,
        max_retries_per_window: int = 10,
        window_seconds: float = 60.0,
    ) -> None:
        if max_retries_per_window < 1:
            raise ValueError("max_retries_per_window must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max = max_retries_per_window
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to consume one retry token.

        Returns ``True`` if a token was available, ``False`` if the budget
        is exhausted.
        """
        async with self._lock:
            now = _time.monotonic()
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self._max:
                return False
            self._timestamps.append(now)
            return True

    async def available_async(self) -> int:
        """Return the number of retry tokens currently available (async-safe)."""
        async with self._lock:
            now = _time.monotonic()
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return max(0, self._max - len(self._timestamps))

    @property
    def available(self) -> int:
        """Number of retry tokens currently available.

        .. note::
            This is a best-effort synchronous read without locking.
            Use :meth:`available_async` for an accurate, lock-protected
            count in concurrent contexts.
        """
        now = _time.monotonic()
        cutoff = now - self._window
        active = sum(1 for t in self._timestamps if t > cutoff)
        return max(0, self._max - active)
