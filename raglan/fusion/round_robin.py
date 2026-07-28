"""Round-robin fusion — interleaves results from each retriever.

Useful when you want equal representation from each retriever rather
than relying on score magnitudes.
"""

from __future__ import annotations

from typing import Any

from raglan.types import ScoredChunk


class RoundRobinFusion:
    """Round-robin interleaving of multiple retrievers.

    Takes the first result from each retriever, then the second, etc.
    De-duplication is handled on-the-fly — a chunk that already appeared
    from an earlier retriever is skipped.
    """

    name = "round_robin"

    async def fuse(
        self,
        retriever_results: dict[str, list[list[ScoredChunk]]],
        original_query_idx: int = 0,
    ) -> list[ScoredChunk]:
        seen: set[str] = set()
        seen_parents: set[str] = set()
        fused: list[ScoredChunk] = []

        # Flatten each retriever's results across all query variants
        per_retriever: list[list[ScoredChunk]] = []
        for query_results in retriever_results.values():
            flat: list[ScoredChunk] = []
            for qr in query_results:
                flat.extend(qr)
            per_retriever.append(flat)

        max_len = max((len(r) for r in per_retriever), default=0)
        for i in range(max_len):
            for retriever_list in per_retriever:
                if i < len(retriever_list):
                    chunk = retriever_list[i]
                    if chunk.chunk_id in seen:
                        continue
                    pid = chunk.parent_chunk_id or chunk.chunk_id
                    if pid in seen_parents:
                        continue
                    seen.add(chunk.chunk_id)
                    seen_parents.add(pid)
                    # Tag chunk as fused while preserving original source info
                    chunk.source = f"{chunk.source}→fused"
                    fused.append(chunk)

        return fused

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "params": {}}
