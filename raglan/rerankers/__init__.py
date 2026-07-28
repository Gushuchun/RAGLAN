"""Rerankers — fine-grained relevance re-ranking of candidate documents."""

from raglan.rerankers.cross_encoder import CrossEncoderReranker

try:
    from raglan.rerankers.cohere import CohereReranker
except ImportError:  # pragma: no cover — cohere is optional
    CohereReranker = None  # type: ignore[assignment,misc]

__all__ = ["CrossEncoderReranker"]

if CohereReranker is not None:
    __all__.append("CohereReranker")
