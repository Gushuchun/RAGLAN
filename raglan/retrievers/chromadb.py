"""ChromaDB dense retriever.

Requires ``pip install raglan-retrieval[chromadb]`` (or ``chromadb``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from raglan.exceptions import FilterError
from raglan.types import Filter, Op, ScoredChunk

logger = logging.getLogger(__name__)


class ChromaDBRetriever:
    """Dense retriever backed by ChromaDB.

    Parameters
    ----------
    collection_name:
        ChromaDB collection to query.
    client:
        Optional pre-configured ``chromadb.Client`` or
        ``chromadb.AsyncClient``.  When ``None``, an in-memory client is
        created.
    persist_directory:
        Directory to persist the ChromaDB database.  Only used when
        *client* is ``None`` and ``chromadb.PersistentClient`` is created.
    embedding_function:
        Optional ChromaDB embedding function.  When ``None``, embeddings
        must be provided at query time.  When set, ChromaDB handles
        embedding internally and the pipeline Embedder stage can be
        omitted for this retriever.
    distance_metric:
        ChromaDB distance metric (``"cosine"``, ``"l2"``, ``"ip"``).
    """

    name = "chromadb"
    requires_embeddings = True

    _DISTANCE_MAP: ClassVar[dict[str, str]] = {
        "cosine": "cosine",
        "l2": "l2",
        "ip": "ip",
    }

    def __init__(
        self,
        collection_name: str = "raglan",
        *,
        client: Any = None,
        persist_directory: str | None = None,
        embedding_function: Any = None,
        distance_metric: str = "cosine",
    ) -> None:
        if distance_metric not in self._DISTANCE_MAP:
            raise ValueError(
                f"distance_metric must be one of "
                f"{list(self._DISTANCE_MAP)}, got '{distance_metric}'"
            )

        self._collection_name = collection_name
        self._client = client
        self._own_client = client is None  # True if we created the client
        self._persist_dir = persist_directory
        self._embedding_fn = embedding_function
        self._distance = distance_metric
        self._collection: Any = None

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
    ) -> list[list[ScoredChunk]]:
        col = self._get_collection()
        results: list[list[ScoredChunk]] = []

        chroma_filter = self._build_chroma_filter(filters) if filters else None

        for emb in embeddings:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [emb],
                "n_results": top_k,
            }
            if chroma_filter is not None:
                query_kwargs["where"] = chroma_filter

            resp = await asyncio.to_thread(col.query, **query_kwargs)

            chunks: list[ScoredChunk] = []
            if resp["ids"] and resp["ids"][0]:
                for i, doc_id in enumerate(resp["ids"][0]):
                    raw_dist = (
                        resp["distances"][0][i]
                        if resp["distances"] and resp["distances"][0]
                        else 0.0
                    )
                    score = _distance_to_score(raw_dist, self._distance)
                    chunks.append(
                        ScoredChunk(
                            chunk_id=doc_id,
                            content=resp["documents"][0][i]
                            if resp["documents"] and resp["documents"][0]
                            else "",
                            score=score,
                            chunk_metadata=resp["metadatas"][0][i]
                            if resp["metadatas"] and resp["metadatas"][0]
                            else {},
                            source=self.name,
                        )
                    )
            results.append(chunks)

        return results

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None]]],
    ) -> None:
        col = self._get_collection()
        # Clear existing data
        try:
            existing = col.get()
            if existing.get("ids"):
                col.delete(ids=existing["ids"])
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning("Could not clear existing ChromaDB collection data: %s", e)
        except Exception:
            logger.warning(
                "Unexpected error clearing ChromaDB collection '%s'", self._collection_name
            )
            raise

        async for batch in chunks:
            ids = []
            documents = []
            metadatas: list[dict[str, Any] | None] = []
            embeddings: list[list[float]] = []
            has_metadata = False
            for item in batch:
                cid, content = item[0], item[1]
                meta = item[2] if len(item) > 2 else None
                emb = item[3] if len(item) > 3 else None
                ids.append(cid)
                documents.append(content)
                if meta:
                    has_metadata = True
                    metadatas.append(meta)
                else:
                    metadatas.append(None)
                if emb is not None:
                    embeddings.append(emb)  # type: ignore[unreachable]

            add_kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": documents,
            }
            if has_metadata:
                add_kwargs["metadatas"] = metadatas
            if embeddings:
                add_kwargs["embeddings"] = embeddings
            col.add(**add_kwargs)

    async def add(self, chunks: list[tuple[str, str, dict[str, Any] | None]]) -> None:
        col = self._get_collection()
        ids = []
        documents = []
        metadatas: list[dict[str, Any] | None] = []
        embeddings: list[list[float]] = []
        has_metadata = False
        for item in chunks:
            cid, content = item[0], item[1]
            meta = item[2] if len(item) > 2 else None
            emb = item[3] if len(item) > 3 else None
            ids.append(cid)
            documents.append(content)
            if meta:
                has_metadata = True
                metadatas.append(meta)
            else:
                metadatas.append(None)
            if emb is not None:
                embeddings.append(emb)  # type: ignore[unreachable]
        add_kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
        }
        if has_metadata:
            add_kwargs["metadatas"] = metadatas
        if embeddings:
            add_kwargs["embeddings"] = embeddings
        col.add(**add_kwargs)

    async def close(self) -> None:
        """Close the ChromaDB client and release resources."""
        self._collection = None
        if self._own_client and self._client is not None:
            with contextlib.suppress(Exception):
                self._client.clear_system_cache()
            self._client = None

    async def remove(self, chunk_ids: list[str]) -> None:
        col = self._get_collection()
        col.delete(ids=chunk_ids)

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        from raglan._lazy import _import_module

        _import_module("chromadb", hint="pip install raglan-retrieval[chromadb]")
        import chromadb

        if self._client is not None:
            client = self._client
        elif self._persist_dir is not None:
            client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            client = chromadb.Client()

        try:
            self._collection = client.get_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,
            )
        except (ValueError, AttributeError, chromadb.errors.NotFoundError):
            # Collection not found or client type doesn't support get_collection
            logger.debug("Collection '%s' not found, creating new one", self._collection_name)
            self._collection = client.create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": self._distance},
            )
        return self._collection

    @staticmethod
    def _build_chroma_filter(filters: list[Filter]) -> dict[str, Any] | None:
        """Translate Raglan Filter tree to a ChromaDB metadata filter dict.

        Supports all filter operators that ChromaDB understands:
        ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``,
        ``$and``, ``$or``.  Unsupported operators (``exists``, ``contains``)
        are silently ignored.
        """
        if not filters:
            return None

        def _leaf(f: Filter) -> dict[str, Any]:
            key = f.field or ""
            if f.op == Op.EQ:
                return {key: {"$eq": f.value}}
            elif f.op == Op.NE:
                return {key: {"$ne": f.value}}
            elif f.op == Op.GT:
                return {key: {"$gt": f.value}}
            elif f.op == Op.GTE:
                return {key: {"$gte": f.value}}
            elif f.op == Op.LT:
                return {key: {"$lt": f.value}}
            elif f.op == Op.LTE:
                return {key: {"$lte": f.value}}
            elif f.op == Op.IN:
                return {key: {"$in": f.value if isinstance(f.value, list) else [f.value]}}
            elif f.op == Op.EXISTS:
                raise FilterError(
                    "EXISTS filter operator is not supported by ChromaDB. "
                    "Use EQ/NE/GT/GTE/LT/LTE/IN operators."
                )
            elif f.op == Op.CONTAINS:
                raise FilterError(
                    "CONTAINS filter operator is not supported by ChromaDB. "
                    "Use EQ/IN operators instead."
                )
            raise FilterError(f"Unsupported filter operator for ChromaDB: {f.op}")

        def _walk(fs: list[Filter]) -> dict[str, Any] | None:
            if len(fs) == 1:
                f = fs[0]
                if f.op in (Op.AND, Op.OR):
                    joiner = "$and" if f.op == Op.AND else "$or"
                    if not f.children:
                        return None
                    children = [_walk([c]) for c in f.children]
                    return {joiner: [c for c in children if c is not None]}
                return _leaf(f)

            children = [_walk([f]) for f in fs]
            return {"$and": [c for c in children if c is not None]}

        return _walk(filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "collection_name": self._collection_name,
                "persist_directory": self._persist_dir,
                "distance_metric": self._distance,
            },
        }


def _distance_to_score(distance: float, metric: str) -> float:
    """Convert a ChromaDB distance to a [0, 1] relevance score.

    - **cosine**: distance ∈ [0, 2] → score = 1 - distance/2
    - **l2**: distance ∈ [0, ∞) → score = 1 / (1 + distance)
    - **ip** (inner product): distance is already similarity-like
    """
    if metric == "cosine":
        return 1.0 - distance / 2.0
    elif metric == "l2":
        return 1.0 / (1.0 + distance)
    elif metric == "ip":
        return max(0.0, min(1.0, distance))
    return 1.0 - distance
