"""Shared utilities for query expanders — no circular imports."""

from __future__ import annotations

from typing import Any

# Shared prompt template for LLM-based query expanders.
# Must include ``{query}`` and ``{num_variants}`` placeholders.
DEFAULT_EXPANDER_PROMPT = """\
Generate {num_variants} alternative search queries that express the same
information need in different words. Cover synonyms, related terms, and
different perspectives.

Original query: {query}

Return JSON only: {{"variants": ["variant1", "variant2", ...]}}"""


def ensure_original_query_first(query: str, variants: list[str]) -> list[str]:
    """Return *variants* with *query* guaranteed as the first element."""
    if query not in variants:
        variants.insert(0, query)
    elif variants[0] != query:
        variants.remove(query)
        variants.insert(0, query)
    return variants


def parse_expander_response(
    response_content: str | None, query: str, num_variants: int
) -> tuple[list[str], dict[str, Any]]:
    """Parse the JSON response from an LLM expander call.

    Handles ``JSONDecodeError`` gracefully (falls back to the original
    query), ensures the original query is first, and limits to the
    requested number of variants.
    """
    import json as _json

    try:
        data = _json.loads(response_content or "{}")
        variants: list[str] = data.get("variants", [query])
    except _json.JSONDecodeError:
        variants = [query]

    variants = ensure_original_query_first(query, variants)
    return variants[: 1 + num_variants], {}
