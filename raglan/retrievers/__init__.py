"""Retriever implementations — dense and sparse search backends."""

from raglan.retrievers.bm25 import BM25Retriever
from raglan.retrievers.memory import MemoryRetriever

try:
    from raglan.retrievers.configurable_pgvector import (
        ConfigurablePgvectorRetriever,
    )
except ImportError:  # pragma: no cover — asyncpg is optional
    ConfigurablePgvectorRetriever = None  # type: ignore[assignment,misc]

try:
    from raglan.retrievers.chromadb import ChromaDBRetriever
except ImportError:  # pragma: no cover — chromadb is optional
    ChromaDBRetriever = None  # type: ignore[assignment,misc]

try:
    from raglan.retrievers.qdrant import QdrantRetriever
except ImportError:  # pragma: no cover — qdrant-client is optional
    QdrantRetriever = None  # type: ignore[assignment,misc]

__all__ = ["BM25Retriever", "MemoryRetriever"]

if ConfigurablePgvectorRetriever is not None:
    __all__.append("ConfigurablePgvectorRetriever")
if ChromaDBRetriever is not None:
    __all__.append("ChromaDBRetriever")
if QdrantRetriever is not None:
    __all__.append("QdrantRetriever")
