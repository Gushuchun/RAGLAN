"""Qdrant dense retriever.

Requires ``pip install raglan-retrieval[qdrant]`` (or ``qdrant-client``).
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from raglan.types import Filter, Op, ScoredChunk

# Deterministic UUID namespace for mapping chunk_ids to Qdrant point IDs.
_QDRANT_UUID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _cid_to_point_id(cid: str) -> str:
    """Return a deterministic UUID string for the given chunk_id."""
    return str(uuid.uuid5(_QDRANT_UUID_NS, cid))


class QdrantRetriever:
    """Dense retriever backed by Qdrant vector database.

    Parameters
    ----------
    collection_name:
        Qdrant collection to query.
    url:
        Qdrant server URL (e.g. ``"http://localhost:6333"``).  When
        ``None``, an in-memory client is used.
    api_key:
        Optional API key for Qdrant Cloud.
    host:
        Optional host for gRPC mode.  Use with *port*.
    port:
        Optional gRPC port.  Default 6334.
    vector_name:
        Name of the named vector in Qdrant.  When ``None``, the default
        unnamed vector is used.
    distance_metric:
        Qdrant distance metric (``"cosine"``, ``"euclid"``, ``"dot"``).
    prefer_grpc:
        When ``True``, prefer gRPC transport (requires ``qdrant-client[grpc]``).
    """

    name = "qdrant"
    requires_embeddings = True

    _DISTANCE_MAP: ClassVar[dict[str, str]] = {
        "cosine": "COSINE",
        "euclid": "EUCLID",
        "dot": "DOT",
    }

    def __init__(
        self,
        collection_name: str = "raglan",
        *,
        url: str | None = None,
        api_key: str | None = None,
        host: str | None = None,
        port: int | None = None,
        vector_name: str | None = None,
        distance_metric: str = "cosine",
        prefer_grpc: bool = False,
        vector_size: int = 0,
    ) -> None:
        if distance_metric not in self._DISTANCE_MAP:
            raise ValueError(
                f"distance_metric must be one of "
                f"{list(self._DISTANCE_MAP)}, got '{distance_metric}'"
            )

        self._collection_name = collection_name
        self._url = url
        self._api_key = api_key
        self._host = host
        self._port = port
        self._vector_name = vector_name
        self._distance = distance_metric
        self._prefer_grpc = prefer_grpc
        self._vector_size = vector_size
        self._client: Any = None
        self._client_lock = asyncio.Lock()
        self._initialised = False

    async def retrieve(
        self,
        queries: list[str],
        embeddings: list[list[float]],
        top_k: int,
        filters: list[Filter] | None = None,
        timeout: float | None = None,
    ) -> list[list[ScoredChunk]]:
        first_emb = embeddings[0] if embeddings else []
        await self._ensure_collection(vector_dim=len(first_emb))
        results: list[list[ScoredChunk]] = []

        qdrant_filter = self._build_qdrant_filter(filters) if filters else None

        for emb in embeddings:
            query_vector: Any = {self._vector_name: emb} if self._vector_name else emb
            try:
                # qdrant-client >= 1.13: unified Query API
                from qdrant_client.models import NearestQuery

                query_args: dict[str, Any] = {
                    "collection_name": self._collection_name,
                    "query": NearestQuery(nearest=query_vector),
                    "limit": top_k,
                    "with_payload": True,
                }
            except ImportError:
                # Fallback for older qdrant-client versions
                query_args = {
                    "collection_name": self._collection_name,
                    "query_vector": query_vector,
                    "limit": top_k,
                    "with_payload": True,
                }
            if qdrant_filter is not None:
                query_args["query_filter"] = qdrant_filter

            resp = await self._client.query_points(**query_args)

            chunks: list[ScoredChunk] = []
            for point in resp.points:
                payload = point.payload or {}
                chunks.append(
                    ScoredChunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        content=payload.get("content", ""),
                        score=point.score or 0.0,
                        parent_chunk_id=payload.get("parent_chunk_id"),
                        chunk_metadata=payload.get("metadata", {}),
                        source=self.name,
                    )
                )
            results.append(chunks)

        return results

    async def index(
        self,
        chunks: AsyncIterator[list[tuple[str, str, dict[str, Any] | None, list[float] | None]]],
    ) -> None:
        # Pre-fetch first batch to detect vector dimension
        first: list[tuple[str, str, dict[str, Any] | None, list[float] | None]] | None = None
        async for batch in chunks:
            first = batch
            break

        first_dim = 0
        if first is not None:
            for item in first:
                emb = item[3] if len(item) > 3 else None
                if emb is not None:
                    first_dim = len(emb)
                    break

        # Recreate collection with the correct vector size
        if self._client is None:
            await self._ensure_client()
        with contextlib.suppress(ValueError, AttributeError):
            await self._client.delete_collection(self._collection_name)
        self._initialised = False
        await self._ensure_collection(vector_dim=first_dim)

        # Index first batch (already consumed from iterator) + remaining batches
        async def _upsert_batch(
            batch: list[tuple[str, str, dict[str, Any] | None, list[float] | None]],
        ) -> None:
            from qdrant_client.models import PointStruct

            points: list[PointStruct] = []
            for item in batch:
                cid, content = item[0], item[1]
                meta = item[2] if len(item) > 2 else None
                emb = item[3] if len(item) > 3 else None
                pid = _cid_to_point_id(cid)
                payload_dict: dict[str, Any] = {
                    "content": content,
                    "chunk_id": cid,
                    "parent_chunk_id": meta.get("parent_chunk_id") if meta else None,
                    "metadata": meta or {},
                }
                vector: dict[str, list[float]] | list[float] = (
                    {self._vector_name: emb}
                    if self._vector_name and emb is not None
                    else emb
                    if emb is not None
                    else []
                )
                points.append(PointStruct(id=pid, vector=vector, payload=payload_dict))  # type: ignore
            if points:
                await self._client.upsert(collection_name=self._collection_name, points=points)

        if first is not None:
            await _upsert_batch(first)
        async for batch in chunks:
            await _upsert_batch(batch)

    async def add(
        self,
        chunks: list[tuple[str, str, dict[str, Any] | None, list[float] | None]],
    ) -> None:
        # Detect vector size from first chunk with embedding
        first_dim = 0
        for item in chunks:
            emb = item[3] if len(item) > 3 else None
            if emb is not None:
                first_dim = len(emb)
                break
        await self._ensure_collection(vector_dim=first_dim)

        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for item in chunks:
            cid, content = item[0], item[1]
            meta = item[2] if len(item) > 2 else None
            emb = item[3] if len(item) > 3 else None
            pid = _cid_to_point_id(cid)
            payload_dict: dict[str, Any] = {
                "content": content,
                "chunk_id": cid,
                "parent_chunk_id": meta.get("parent_chunk_id") if meta else None,
                "metadata": meta or {},
            }
            vector: dict[str, list[float]] | list[float] = (
                {self._vector_name: emb}
                if self._vector_name and emb is not None
                else emb
                if emb is not None
                else []
            )
            points.append(PointStruct(id=pid, vector=vector, payload=payload_dict))  # type: ignore

        if points:
            await self._client.upsert(collection_name=self._collection_name, points=points)

    async def remove(self, chunk_ids: list[str]) -> None:
        await self._ensure_collection()
        if chunk_ids:
            point_ids = [_cid_to_point_id(cid) for cid in chunk_ids]
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=point_ids,
            )

    async def close(self) -> None:
        """Close the Qdrant client and release all resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._initialised = False

    async def _ensure_collection(self, vector_dim: int = 0) -> None:
        if self._initialised:
            return
        if self._client is None:
            await self._ensure_client()

        from qdrant_client.models import Distance, VectorParams

        # Determine vector size: explicit > embedding > fallback
        dim = self._vector_size if self._vector_size > 0 else vector_dim
        if dim <= 0:
            dim = 1536

        # Ensure collection exists
        try:
            await self._client.get_collection(self._collection_name)
        except Exception:
            # qdrant-client raises ValueError/AttributeError (v0.x) or
            # UnexpectedResponse (v1.x) when the collection is missing.
            # We catch broadly and create the collection.
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance[self._DISTANCE_MAP[self._distance]],
                ),
            )

        self._initialised = True

    async def _ensure_client(self) -> None:
        """Lazily create the Qdrant client (concurrency-safe)."""
        if self._client is not None:
            return
        async with self._client_lock:
            if self._client is not None:  # double-check
                return  # type: ignore[unreachable]
            from raglan._lazy import _import_module

            _import_module("qdrant_client", hint="pip install raglan-retrieval[qdrant]")
            from qdrant_client import AsyncQdrantClient

            if self._url is not None:
                self._client = AsyncQdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc,
                )
            elif self._host is not None:
                self._client = AsyncQdrantClient(
                    host=self._host,
                    port=self._port or 6334,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc,
                )
            else:
                self._client = AsyncQdrantClient(location=":memory:")

    @staticmethod
    def _build_qdrant_filter(filters: list[Filter]) -> Any | None:
        """Translate Raglan Filter tree to a Qdrant Filter object.

        Supports nested AND / OR via ``must`` / ``should`` clauses.
        Leaf conditions use ``FieldCondition`` for equality and range
        comparisons.
        """
        if not filters:
            return None

        try:
            from qdrant_client.models import FieldCondition, MatchValue, Range
            from qdrant_client.models import Filter as QFilter
        except ImportError:
            return None

        def _walk(f: Filter) -> QFilter:
            # Compound: AND → must, OR → should
            if f.op in (Op.AND, Op.OR):
                children: list[Any] = [_walk(c) for c in (f.children or [])]
                if f.op == Op.AND:
                    return QFilter(must=children)
                return QFilter(should=children)

            # Leaf conditions — metadata is stored in a nested "metadata" key
            key = f"metadata.{f.field}" if f.field else ""
            conditions: list[FieldCondition] = []
            if f.op == Op.EQ:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=f.value)))
            elif f.op == Op.GT:
                conditions.append(FieldCondition(key=key, range=Range(gt=f.value)))
            elif f.op == Op.GTE:
                conditions.append(FieldCondition(key=key, range=Range(gte=f.value)))
            elif f.op == Op.LT:
                conditions.append(FieldCondition(key=key, range=Range(lt=f.value)))
            elif f.op == Op.LTE:
                conditions.append(FieldCondition(key=key, range=Range(lte=f.value)))
            elif f.op == Op.NE:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=f.value)))
                return QFilter(must_not=conditions)  # type: ignore
            return QFilter(must=conditions)  # type: ignore

        if len(filters) == 1:
            return _walk(filters[0])
        return QFilter(must=[_walk(f) for f in filters])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "params": {
                "collection_name": self._collection_name,
                "url": self._url,
                "host": self._host,
                "port": self._port,
                "vector_name": self._vector_name,
                "distance_metric": self._distance,
            },
        }
