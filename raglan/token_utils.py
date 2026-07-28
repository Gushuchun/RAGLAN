"""Token counting utility using tiktoken with a character-based fallback.

``tiktoken`` is an optional dependency.  When installed the exact
tokeniser used by OpenAI models (``cl100k_base``) is employed.
Otherwise a heuristic that distinguishes ASCII from CJK text provides
a reasonable estimate.
"""

from __future__ import annotations

from typing import Any


def _detect_cjk_ratio(text: str) -> float:
    """Return the fraction of characters that fall in the CJK range."""
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")
    return cjk / len(text)


_ENCODING: Any = None


def _get_encoding() -> Any:
    """Lazily load the tiktoken encoding (cl100k_base)."""
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            _ENCODING = False  # sentinel — use fallback
    return _ENCODING if _ENCODING is not False else None


def count_tokens(text: str, model: str | None = None) -> int:
    """Return the number of tokens in *text*.

    Parameters
    ----------
    text:
        The text to count tokens for.
    model:
        Optional model name for model-specific encoding.  When ``None``
        (the default) the ``cl100k_base`` encoding is used, which covers
        GPT-4, GPT-3.5-turbo, and text-embedding-3-* models.

    Returns
    -------
    int
        Token count (always >= 1 for non-empty strings).
    """
    if not text:
        return 0

    enc = _get_encoding()
    if enc is not None:
        if model is not None:
            try:
                import tiktoken

                enc = tiktoken.encoding_for_model(model)
            except (KeyError, ImportError):
                pass  # fall through to default encoding
        return len(enc.encode(text))

    # --- fallback heuristic ------------------------------------------------
    # Split into segments so we can apply different ratios:
    #   - CJK characters:   ~1.5 chars / token
    #   - ASCII / other:    ~4.0 chars / token
    cjk_ratio = _detect_cjk_ratio(text)
    chars_per_token = 1.5 + cjk_ratio * 2.0 if cjk_ratio > 0.5 else 4.0 - cjk_ratio * 2.5

    return max(1, int(len(text) / chars_per_token))


def estimate_tokens(text: str) -> int:
    """Deprecated alias — use :func:`count_tokens` instead."""
    return count_tokens(text)
