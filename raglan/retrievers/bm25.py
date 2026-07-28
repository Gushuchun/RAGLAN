"""Pure-Python BM25 retriever with zero external dependencies.

Implements Okapi BM25 (``k1=1.5``, ``b=0.75``) with a built-in tokenizer
that handles both English (whitespace) and Chinese (bigram) text.
All mutation methods are protected by an ``asyncio.Lock`` for safe
concurrent use in production deployments.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from typing import Any

from raglan.types import Filter, ScoredChunk


class BM25Retriever:
    """BM25 sparse retriever backed by an in-memory inverted index.

    Suitable for corpora up to ~1 million documents on commodity hardware.
    For larger collections, implement the ``Retriever`` protocol against
    Elasticsearch or a similar search engine.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter. Higher values give more
        weight to repeated terms within a document.
    b:
        Document-length normalisation parameter. ``0`` disables length
        normalisation; ``1`` applies full normalisation.
    tokenizer:
        Optional callable ``(text: str) -> list[str]``. When ``None``,
        the built-in Chinese-bigram / English-whitespace tokenizer is used.
    stopwords:
        Optional set of lowercase words to exclude from the index.
        When ``None`` a small built-in English stopword set is used.
    """

    name = "bm25"
    requires_embeddings = False

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
        # Average document length in tokens
        self._avgdl: float = 0.0
        # Total document count
        self._doc_count: int = 0

    # ------------------------------------------------------------------
    # Public API — Retriever protocol
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
    ) -> list[list[ScoredChunk]]:
        """Search for *top_k* chunks per query.

        *embeddings*, *filters*, and *timeout* are accepted for protocol
        compatibility but are not used by BM25.
        """
        if filters:
            logging.getLogger(__name__).warning(
                "BM25Retriever does not support metadata filters — "
                "filters will be ignored. Use a dense retriever for "
                "filtered search."
            )
        async with self._lock:
            results: list[list[ScoredChunk]] = []
            for query in queries:
                query_tokens = [t for t in self._tokenizer(query) if t not in self._stopwords]
                scores: dict[str, float] = {}

                for token in set(query_tokens):
                    idf = self._idf(token)
                    if idf == 0.0:
                        continue
                    for doc_id, tf in self._inverted.get(token, {}).items():
                        dl = len(self._docs[doc_id][1])
                        score = idf * self._tf_component(tf, dl)
                        scores[doc_id] = scores.get(doc_id, 0.0) + score

                top_items = heapq.nlargest(
                    min(top_k, len(scores)), scores.items(), key=lambda x: x[1]
                )
                results.append(
                    [
                        ScoredChunk(
                            chunk_id=doc_id,
                            content=self._docs[doc_id][0],
                            score=scores[doc_id],
                            source=self.name,
                        )
                        for doc_id, _ in top_items
                    ]
                )

            return results

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        """Build the inverted index from a stream of chunk batches.

        Uses a double-buffer strategy: the new index is built off to the
        side and swapped in atomically, so concurrent ``retrieve()`` calls
        continue to see the old index.
        """
        new_docs: dict[str, tuple[str, dict[str, int]]] = {}
        new_inverted: dict[str, dict[str, int]] = defaultdict(dict)
        total_tokens = 0
        doc_count = 0

        async for batch in chunks:
            for item in batch:
                chunk_id, content = item[0], item[1]
                tokens = [t for t in self._tokenizer(content) if t not in self._stopwords]
                tf = self._count_terms(tokens)
                new_docs[chunk_id] = (content, tf)
                total_tokens += len(tokens)
                doc_count += 1
                for term, freq in tf.items():
                    new_inverted[term][chunk_id] = freq

        # Atomically swap
        async with self._lock:
            self._docs = new_docs
            self._inverted = new_inverted
            self._avgdl = total_tokens / max(1, doc_count)
            self._doc_count = doc_count

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        """Incrementally add chunks to an existing index."""
        async with self._lock:
            total_dl = self._avgdl * self._doc_count
            for item in chunks:
                chunk_id, content = item[0], item[1]
                tokens = [t for t in self._tokenizer(content) if t not in self._stopwords]
                tf = self._count_terms(tokens)
                self._docs[chunk_id] = (content, tf)
                total_dl += len(tokens)
                self._doc_count += 1
                for term, freq in tf.items():
                    self._inverted[term][chunk_id] = freq
            self._avgdl = total_dl / max(1, self._doc_count)

    async def remove(self, chunk_ids: list[str]) -> None:
        """Incrementally remove chunks from the index."""
        async with self._lock:
            total_dl = self._avgdl * self._doc_count
            for cid in chunk_ids:
                if cid not in self._docs:
                    continue
                _content, tf = self._docs.pop(cid)
                total_dl -= sum(tf.values())
                self._doc_count -= 1
                for term in tf:
                    self._inverted[term].pop(cid, None)
                    if not self._inverted[term]:
                        del self._inverted[term]
            self._avgdl = total_dl / max(1, self._doc_count)

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        doc_freq = len(self._inverted.get(term, {}))
        if doc_freq == 0:
            return 0.0
        return math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _tf_component(self, tf: int, doc_len: int) -> float:
        numerator = tf * (self._k1 + 1.0)
        denominator = tf + self._k1 * (1.0 - self._b + self._b * doc_len / max(1.0, self._avgdl))
        return numerator / denominator

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _count_terms(tokens: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    @staticmethod
    def _builtin_tokenizer(text: str) -> list[str]:
        """Whitespace-split for English, bigram for CJK characters."""
        tokens: list[str] = []
        for word in text.lower().split():
            if any("一" <= c <= "鿿" for c in word):
                # CJK Unified Ideographs: bigram + whole word
                for i in range(len(word) - 1):
                    tokens.append(word[i : i + 2])
                tokens.append(word)
            else:
                tokens.append(word)
        return tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "k1": self._k1,
                "b": self._b,
            },
        }
