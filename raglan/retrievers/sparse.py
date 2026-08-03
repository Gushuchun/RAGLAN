"""Sparse (keyword) search index abstraction.

``BM25Retriever`` delegates indexing and scoring to a :class:`SparseIndex`.
The default is the pure-Python :class:`MemorySparseIndex` (Okapi BM25 with a
CJK-aware tokenizer).  Implement the :class:`SparseIndex` protocol to plug in
an external engine — Elasticsearch, meilisearch, OpenSearch, etc.
"""

from __future__ import annotations

import asyncio
import heapq
import math
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from raglan.types import ScoredChunk


class SparseIndex(Protocol):
    """Storage + retrieval for keyword (sparse) search.

    A backend owns its inverted index, tokenization, and scoring.  The
    retriever hands it query strings and raw chunks.
    """

    name: str

    async def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return the top-*top_k* chunks for *query*, ranked by relevance."""
        ...

    async def index(
        self, chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]]
    ) -> None:
        """Replace the index contents from a stream of chunk batches."""
        ...

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Incrementally add chunks to the index."""
        ...

    async def remove(self, chunk_ids: list[str]) -> None:
        """Incrementally remove chunks from the index."""
        ...


class MemorySparseIndex:
    """Pure-Python in-memory BM25 index — the default SparseIndex backend.

    Implements Okapi BM25 (``k1=1.5``, ``b=0.75``) with a built-in tokenizer
    that handles both English (whitespace) and Chinese (jieba when installed,
    else bigram) text.  Suitable for corpora up to ~1 million documents.

    ``search()`` reads an immutable snapshot of the index and computes without
    holding the lock, so concurrent searches do not serialise.  Writers
    (``index``/``add``/``remove``) mutate under ``asyncio.Lock``.
    """

    name = "memory_bm25"

    _DEFAULT_STOPWORDS: frozenset[str] = frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "our",
            "their",
            "not",
            "no",
            "nor",
            "so",
            "if",
            "then",
            "than",
            "too",
            "very",
            "just",
            "about",
            "also",
        }
    )

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
        stopwords: set[str] | None = None,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._tokenizer = tokenizer or self._builtin_tokenizer
        self._stopwords = stopwords if stopwords is not None else self._DEFAULT_STOPWORDS
        self._lock = asyncio.Lock()

        # Per-document data: chunk_id -> (content, term-frequency dict)
        self._docs: dict[str, tuple[str, dict[str, int]]] = {}
        # Inverted index: term -> {chunk_id: term_frequency}
        self._inverted: dict[str, dict[str, int]] = defaultdict(dict)
        # Per-document metadata: chunk_id -> metadata dict
        self._meta: dict[str, dict[str, Any]] = {}
        # Integer running totals — avgdl is derived to avoid float drift.
        self._total_tokens: int = 0
        self._doc_count: int = 0

    @property
    def _avgdl(self) -> float:
        """Average document length in tokens (derived, drift-free)."""
        return self._total_tokens / max(1, self._doc_count)

    # ------------------------------------------------------------------
    # SparseIndex protocol
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Search for *top_k* chunks matching *query*."""
        # Snapshot references, then compute synchronously.  The body below has
        # no await points, so it runs atomically in the event loop — no lock
        # needed, and concurrent searches no longer serialise.  Writers mutate
        # under _lock or swap references atomically, which this snapshot sees
        # consistently.
        docs = self._docs
        inverted = self._inverted
        meta = self._meta
        avgdl = self._avgdl

        query_tokens = [t for t in self._tokenizer(query) if t not in self._stopwords]
        scores: dict[str, float] = {}

        for token in set(query_tokens):
            idf = self._idf(token)
            if idf == 0.0:
                continue
            for doc_id, tf in inverted.get(token, {}).items():
                dl = len(docs[doc_id][1])
                score = idf * self._tf_component(tf, dl, avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        top_items = heapq.nlargest(min(top_k, len(scores)), scores.items(), key=lambda x: x[1])
        return [
            ScoredChunk(
                chunk_id=doc_id,
                content=docs[doc_id][0],
                score=scores[doc_id],
                chunk_metadata=meta.get(doc_id, {}),
                source=self.name,
            )
            for doc_id, _ in top_items
        ]

    async def index(
        self, chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]]
    ) -> None:
        """Build the inverted index from a stream of chunk batches.

        Uses a double-buffer strategy: the new index is built off to the side
        and swapped in atomically, so concurrent ``search()`` calls continue to
        see the old index.
        """
        new_docs: dict[str, tuple[str, dict[str, int]]] = {}
        new_inverted: dict[str, dict[str, int]] = defaultdict(dict)
        new_meta: dict[str, dict[str, Any]] = {}
        total_tokens = 0
        doc_count = 0

        async for batch in chunks:
            for item in batch:
                chunk_id, content = item[0], item[1]
                meta = item[2] if len(item) > 2 else None
                tokens = [t for t in self._tokenizer(content) if t not in self._stopwords]
                tf = self._count_terms(tokens)
                new_docs[chunk_id] = (content, tf)
                if meta:
                    new_meta[chunk_id] = meta
                total_tokens += len(tokens)
                doc_count += 1
                for term, freq in tf.items():
                    new_inverted[term][chunk_id] = freq

        # Atomically swap
        async with self._lock:
            self._docs = new_docs
            self._inverted = new_inverted
            self._meta = new_meta
            self._total_tokens = total_tokens
            self._doc_count = doc_count

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Incrementally add chunks to an existing index."""
        async with self._lock:
            for item in chunks:
                chunk_id, content = item[0], item[1]
                meta = item[2] if len(item) > 2 else None
                tokens = [t for t in self._tokenizer(content) if t not in self._stopwords]
                tf = self._count_terms(tokens)
                self._docs[chunk_id] = (content, tf)
                if meta:
                    self._meta[chunk_id] = meta
                self._total_tokens += len(tokens)
                self._doc_count += 1
                for term, freq in tf.items():
                    self._inverted[term][chunk_id] = freq

    async def remove(self, chunk_ids: list[str]) -> None:
        """Incrementally remove chunks from the index."""
        async with self._lock:
            for cid in chunk_ids:
                if cid not in self._docs:
                    continue
                _content, tf = self._docs.pop(cid)
                self._meta.pop(cid, None)
                self._total_tokens -= sum(tf.values())
                self._doc_count -= 1
                for term in tf:
                    self._inverted[term].pop(cid, None)
                    if not self._inverted[term]:
                        del self._inverted[term]

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        doc_freq = len(self._inverted.get(term, {}))
        if doc_freq == 0:
            return 0.0
        return math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _tf_component(self, tf: int, doc_len: int, avgdl: float | None = None) -> float:
        numerator = tf * (self._k1 + 1.0)
        denominator = tf + self._k1 * (
            1.0 - self._b + self._b * doc_len / max(1.0, avgdl or self._avgdl)
        )
        return numerator / denominator

    @staticmethod
    def _count_terms(tokens: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    @staticmethod
    def _is_cjk(c: str) -> bool:
        """Check if a character is in the CJK Unified Ideographs block."""
        return "一" <= c <= "鿿"

    @staticmethod
    def _is_cjk_extended(c: str) -> bool:
        """Check if a character is in CJK Extended-A or compatible ranges."""
        return "㐀" <= c <= "䶿" or "豈" <= c <= "﫿"

    @classmethod
    def _tokenize_cjk(cls, segment: str) -> list[str]:
        """Tokenize a CJK segment using jieba if available, else bigram."""
        try:
            import jieba

            return list(jieba.cut(segment))
        except ImportError:
            tokens: list[str] = []
            for i in range(len(segment) - 1):
                tokens.append(segment[i : i + 2])
            tokens.append(segment)
            return tokens

    @classmethod
    def _builtin_tokenizer(cls, text: str) -> list[str]:
        """Auto-detect: split-based for English, jieba/bigram for CJK.

        Mixed text is split by whitespace first, then each segment is
        classified as CJK or Latin and tokenized accordingly.
        """
        tokens: list[str] = []
        for word in text.lower().split():
            if any(cls._is_cjk(c) or cls._is_cjk_extended(c) for c in word):
                tokens.extend(cls._tokenize_cjk(word))
            else:
                tokens.append(word)
        return tokens
