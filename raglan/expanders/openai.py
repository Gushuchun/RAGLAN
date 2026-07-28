"""OpenAI-powered query expander.

Requires ``pip install raglan-retrieval[openai]`` (or ``openai``).
"""

from __future__ import annotations

from typing import Any

from raglan.expanders._shared import DEFAULT_EXPANDER_PROMPT


class OpenAIExpander:
    """Generates query variants using an OpenAI-compatible chat model.

    Parameters
    ----------
    model:
        Model name (``gpt-4o-mini``, ``gpt-4o``, or any compatible endpoint).
    temperature:
        Sampling temperature.  Keep low (0.1-0.3) for deterministic variants.
    prompt_template:
        Custom prompt.  Must include ``{query}`` and ``{num_variants}``
        placeholders.
    base_url:
        Optional base URL for OpenAI-compatible proxies (e.g. LiteLLM).
    api_key:
        Optional API key.  If ``None`` the ``OPENAI_API_KEY`` env var is used.
    """

    name = "openai_expander"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.3,
        prompt_template: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._prompt = prompt_template or DEFAULT_EXPANDER_PROMPT
        self._base_url = base_url
        self._api_key = api_key
        self._client: Any = None

    async def expand(self, query: str, num_variants: int = 3) -> tuple[list[str], dict[str, Any]]:
        from raglan.expanders._shared import parse_expander_response

        client = self._get_client()
        prompt = self._prompt.format(query=query, num_variants=num_variants)

        resp = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        return parse_expander_response(resp.choices[0].message.content, query, num_variants)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from raglan._lazy import _import_module

        _import_module("openai", hint="pip install raglan-retrieval[openai]")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    async def close(self) -> None:
        """Close the underlying OpenAI HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a ``{type, params}`` dict for ``Raglan.from_dict()``.

        Sensitive values (``api_key``) are redacted.
        """
        return {
            "type": self.name,
            "params": {
                "model": self._model,
                "temperature": self._temperature,
                "prompt_template": self._prompt,
                "base_url": self._base_url,
                "api_key": "<redacted>" if self._api_key else None,
            },
        }
