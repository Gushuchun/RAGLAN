"""Query expanders — generate search variants from a user query."""

from raglan.expanders.identity import IdentityExpander
from raglan.expanders.openai import OpenAIExpander

try:
    from raglan.expanders.litellm import LiteLLMExpander
except ImportError:  # pragma: no cover — litellm is optional
    LiteLLMExpander = None  # type: ignore[assignment,misc]

__all__ = ["IdentityExpander", "OpenAIExpander"]

if LiteLLMExpander is not None:
    __all__.append("LiteLLMExpander")
