"""Identity query expander — the default (no-op) expander."""

from __future__ import annotations

from typing import Any


class IdentityExpander:
    """Returns the original query unchanged — no LLM calls, no variants.

    This is the default expander when the user does not supply an
    ``llm_caller``.  It costs nothing and keeps the pipeline simple.
    """

    name = "identity"

    async def expand(self, query: str, num_variants: int = 3) -> tuple[list[str], dict[str, Any]]:
        return ([query], {})
