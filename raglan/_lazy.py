"""Lazy-import utility — reduces boilerplate for optional-dependency imports.

Used by embedders, expanders, rerankers, and retrievers that defer importing
heavy / optional packages until they are first needed.
"""

from __future__ import annotations

import importlib
from typing import Any


class _MissingPackage:
    """Sentinel returned when a package is not installed."""

    pass


def _import_module(module: str, *, hint: str) -> Any:
    """Import *module* or raise a user-friendly ``ImportError``.

    Parameters
    ----------
    module:
        Fully-qualified module name, e.g. ``"openai"`` or
        ``"sentence_transformers"``.
    hint:
        Human-readable install hint, e.g.
        ``"pip install raglan-retrieval[openai]"``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"This component requires the {module!r} package. Install with: {hint}"
        ) from e
