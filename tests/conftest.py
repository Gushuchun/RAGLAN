"""Shared fixtures for Raglan tests."""

from __future__ import annotations


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: test that requires a real backend service")
    config.addinivalue_line(
        "markers", "slow: test that requires model download or heavy computation"
    )
