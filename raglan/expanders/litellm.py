"""LiteLLM-powered query expander — 100+ LLM providers via a single interface.

Requires ``pip install raglan-retrieval[litellm]`` (or ``litellm``).
"""

from __future__ import annotations

from typing import Any

from raglan.expanders._shared import DEFAULT_EXPANDER_PROMPT


class LiteLLMExpander:
    """Generates query variants using any LiteLLM-supported LLM provider.

    Supports OpenAI, Anthropic, Azure, Bedrock, Vertex AI, Ollama,
    Groq, Together AI, and 100+ other providers.

    Parameters
    ----------
    model:
        LiteLLM model string (e.g. ``"gpt-4o-mini"``, ``"claude-3-haiku-20240307"``,
        ``"azure/your-deployment"``, ``"ollama/llama3"``).
    temperature:
        Sampling temperature.  Keep low (0.1-0.3) for deterministic variants.
    prompt_template:
        Custom prompt.  Must include ``{query}`` and ``{num_variants}``
        placeholders.
    api_key:
        Optional API key.  LiteLLM checks env vars by default
        (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
    api_base:
        Optional custom API base URL.
    """

    name = "litellm_expander"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.3,
        prompt_template: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._prompt = prompt_template or DEFAULT_EXPANDER_PROMPT
        self._api_key = api_key
        self._api_base = api_base

    async def expand(self, query: str, num_variants: int = 3) -> tuple[list[str], dict[str, Any]]:
        from raglan.expanders._shared import parse_expander_response

        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError(
                "LiteLLMExpander requires litellm. "
                "Install with: pip install raglan-retrieval[litellm]"
            ) from e

        prompt = self._prompt.format(query=query, num_variants=num_variants)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
        }
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base

        # Try with response_format first; fall back without for providers
        # that don't support it (older Ollama, Gemini, etc.)
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = await acompletion(**kwargs)
        except Exception:
            del kwargs["response_format"]
            resp = await acompletion(**kwargs)

        return parse_expander_response(resp.choices[0].message.content, query, num_variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "model": self._model,
                "temperature": self._temperature,
                "prompt_template": self._prompt,
                "api_key": "<redacted>" if self._api_key else None,
                "api_base": self._api_base,
            },
        }
