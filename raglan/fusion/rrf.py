"""Reciprocal Rank Fusion (RRF) — the default fusion strategy."""

from __future__ import annotations

from typing import Any

from raglan.types import ScoredChunk


class RRFFusion:
    """Reciprocal Rank Fusion with configurable per-source weights.

    RRF is insensitive to absolute score scales, making it safe to
    combine results from dense retrievers (cosine/L2 distance), sparse
    retrievers (BM25 term weight), and external APIs without score
    normalisation.

    Parameters
    ----------
    k:
        Smoothing constant. Larger values reduce the influence of rank
        position. The academic standard is 60.
    dense_weight:
        Total weight assigned to all dense (embedding-based) retrievers.
    sparse_weight:
        Total weight assigned to all sparse (keyword-based) retrievers.
    variant_weight:
        Proportion of weight given to query variants vs. the original
        query.  0 = only original; 1 = only variants.
    """

    name = "rrf"

    def __init__(
        self,
        *,
        k: int = 60,
        dense_weight: float = 0.8,
        sparse_weight: float = 0.2,
        variant_weight: float = 0.3,
    ) -> None:
        self._k = k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._variant_weight = variant_weight

    async def fuse(
        self,
        retriever_results: dict[str, list[list[ScoredChunk]]],
        original_query_idx: int = 0,
    ) -> list[ScoredChunk]:
        """Fuse results from one or more retrievers into a single ranked list."""
        chunk_scores: dict[str, float] = {}
        chunk_content: dict[str, str] = {}
        chunk_parent: dict[str, str] = {}

        # Count retrievers by type for correct per-type weight normalisation.
        sparse_count = sum(1 for name in retriever_results if "bm25" in name.lower())
        dense_count = len(retriever_results) - sparse_count

        for retriever_name, query_results in retriever_results.items():
            is_sparse = "bm25" in retriever_name.lower()
            for qi, results in enumerate(query_results):
                q_weight = (
                    1.0 - self._variant_weight
                    if qi == original_query_idx
                    else self._variant_weight / max(1, len(query_results) - 1)
                )
                for rank, chunk in enumerate(results, start=1):
                    rrf = 1.0 / (self._k + rank)
                    weighted = rrf * q_weight
                    if is_sparse:
                        weighted *= self._sparse_weight / max(1, sparse_count)
                    else:
                        weighted *= self._dense_weight / max(1, dense_count)

                    cid = chunk.chunk_id
                    chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weighted
                    chunk_content[cid] = chunk.content
                    chunk_parent[cid] = chunk.parent_chunk_id or cid

        # Deduplicate by parent chunk — keep only the highest-scoring child
        parent_best: dict[str, tuple[str, float]] = {}
        for cid, score in chunk_scores.items():
            pid = chunk_parent.get(cid, cid)
            if pid not in parent_best or score > parent_best[pid][1]:
                parent_best[pid] = (cid, score)

        ranked = sorted(parent_best.values(), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(
                chunk_id=cid,
                content=chunk_content[cid],
                score=score,
                parent_chunk_id=chunk_parent.get(cid),
                source="fused",
            )
            for cid, score in ranked
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "k": self._k,
                "dense_weight": self._dense_weight,
                "sparse_weight": self._sparse_weight,
                "variant_weight": self._variant_weight,
            },
        }
