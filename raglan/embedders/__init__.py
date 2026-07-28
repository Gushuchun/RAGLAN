"""Embedders — convert text to dense vector embeddings."""

from raglan.embedders.huggingface import HuggingFaceEmbedder
from raglan.embedders.openai import OpenAIEmbedder

try:
    from raglan.embedders.dashscope import DashScopeEmbedder
except ImportError:  # pragma: no cover — dashscope is optional
    DashScopeEmbedder = None  # type: ignore[assignment,misc]

__all__ = ["HuggingFaceEmbedder", "OpenAIEmbedder"]

if DashScopeEmbedder is not None:
    __all__.append("DashScopeEmbedder")
