"""Simple weighted-score fusion — an alternative to RRF."""

from __future__ import annotations

from typing import Any

from raglan.types import ScoredChunk


class WeightedFusion:
    """Weighted-score fusion with min-max normalisation per retriever.

    This is simpler than RRF but requires the retrievers' score
    distributions to overlap meaningfully.  Prefer ``RRFFusion`` for
    heterogeneous retriever mixes.
    """

    name = "weighted"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {}  # retriever_name -> weight (default equal)

    async def fuse(
        self,
        retriever_results: dict[str, list[list[ScoredChunk]]],
        original_query_idx: int = 0,
    ) -> list[ScoredChunk]:
        scores: dict[str, float] = {}
        content: dict[str, str] = {}
        parents: dict[str, str] = {}
        metadata_map: dict[str, dict[str, Any]] = {}

        for name, query_results in retriever_results.items():
            w = self._weights.get(name, 1.0 / max(1, len(retriever_results)))
            # Flatten and min-max normalise
            all_chunks = [c for qr in query_results for c in qr]
            if not all_chunks:
                continue
            s_min = min(c.score for c in all_chunks)
            s_max = max(c.score for c in all_chunks)
            denom = s_max - s_min or 1.0
            for chunk in all_chunks:
                norm = (chunk.score - s_min) / denom
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + norm * w
                content[chunk.chunk_id] = chunk.content
                parents[chunk.chunk_id] = chunk.parent_chunk_id or chunk.chunk_id
                metadata_map[chunk.chunk_id] = chunk.chunk_metadata

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(
                chunk_id=cid,
                content=content[cid],
                score=score,
                parent_chunk_id=parents.get(cid),
                chunk_metadata=metadata_map.get(cid, {}),
                source="fused",
            )
            for cid, score in ranked
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {"weights": self._weights},
        }
