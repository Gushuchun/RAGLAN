"""Automated quality evaluation of e2e search results.

Uses pre-computed fixtures + relevance judgments to compute standard IR metrics:
recall@k, precision@k, MRR, NDCG@k.  Runs against all three backends and prints
a comparison report.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _load_json(name):
    with open(FIXTURES_DIR / name, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# IR Metrics
# ============================================================================


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of first relevant hit."""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG: top k are all relevant
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# Backend-specific search (imports pre-computed embeddings, no model needed)
# ============================================================================


async def _pg_search(pool, embedding, top_k):
    emb_str = "[" + ",".join(f"{x:.8g}" for x in embedding) + "]"
    rows = await pool.fetch(
        "SELECT chunk_id FROM e2e_eval ORDER BY embedding <=> $1::vector LIMIT $2",
        emb_str,
        top_k,
    )
    return [r[0] for r in rows]


def _chroma_search(client, embedding, top_k):
    col = client.get_collection("e2e_eval")
    resp = col.query(query_embeddings=[embedding], n_results=top_k)
    if resp["ids"] and resp["ids"][0]:
        return list(resp["ids"][0])
    return []


async def _qdrant_search(qdrant_path, embedding, top_k):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import NearestQuery

    client = AsyncQdrantClient(path=qdrant_path)
    resp = await client.query_points(
        "e2e_eval",
        query=NearestQuery(nearest=embedding),
        limit=top_k,
        with_payload=True,
    )
    await client.close()
    return [p.payload.get("chunk_id", str(p.id)) for p in resp.points]


# ============================================================================
# Backend setup
# ============================================================================


async def _pg_connect():
    import asyncpg

    for cs in [
        os.environ.get("RAGLAN_PGCONN"),
        "postgresql://postgres:postgres@localhost:5432/raglan_test",
        "postgresql://postgres:123456@localhost:5432/raglan_test",
    ]:
        if cs is None:
            continue
        try:
            return await asyncpg.create_pool(cs, min_size=1, max_size=2, command_timeout=10)
        except Exception:
            pass
    return None


async def _pg_setup(pool, chunks):
    await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await pool.execute("DROP TABLE IF EXISTS e2e_eval")
    await pool.execute("CREATE TABLE e2e_eval (chunk_id TEXT PRIMARY KEY, embedding vector(384))")
    for c in chunks:
        emb_str = "[" + ",".join(f"{x:.8g}" for x in c["embedding"]) + "]"
        await pool.execute("INSERT INTO e2e_eval VALUES ($1, $2::vector)", c["chunk_id"], emb_str)


def _chroma_setup(chunks):
    import chromadb

    client = chromadb.Client()
    with contextlib.suppress(Exception):
        client.delete_collection("e2e_eval")
    col = client.create_collection("e2e_eval", metadata={"hnsw:space": "cosine"})
    ids = [c["chunk_id"] for c in chunks]
    embs = [c["embedding"] for c in chunks]
    for i in range(0, len(ids), 100):
        col.add(ids=ids[i : i + 100], embeddings=embs[i : i + 100])
    return client


async def _qdrant_setup(chunks):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    tmpdir = tempfile.mkdtemp(prefix="raglan_eval_")
    client = AsyncQdrantClient(path=tmpdir)
    await client.create_collection(
        "e2e_eval",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    points = [
        PointStruct(
            id=str(uuid.uuid5(ns, c["chunk_id"])),
            vector=c["embedding"],
            payload={"chunk_id": c["chunk_id"]},
        )
        for c in chunks
    ]
    await client.upsert("e2e_eval", points)
    await client.close()
    return tmpdir


# ============================================================================
# Helpers
# ============================================================================


async def _cleanup_pg(pool):
    await pool.execute("DROP TABLE IF EXISTS e2e_eval")
    await pool.close()


# ============================================================================
# Main evaluation
# ============================================================================


def _evaluate_precomputed(pg_results, k_values=(1, 3, 5, 10)):
    """Compute metrics from pre-fetched pgvector results."""
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for _qid, data in pg_results.items():
        retrieved = data["retrieved"]
        rel = data["relevant"]
        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, rel, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, rel, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, rel, k))
        metrics["mrr"].append(mrr(retrieved, rel))

    return {m: sum(v) / len(v) for m, v in metrics.items() if v}


def _evaluate(name, search_fn, queries, relevance, k_values=(1, 3, 5, 10)):
    """Run all queries through a search function and compute metrics."""
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["mrr"] = []

    for q in queries:
        qid = q["id"]
        rel = set(relevance["judgments"].get(qid, {}).get("relevant", []))
        if not rel:
            continue

        retrieved = search_fn(q["embedding"], max(k_values))

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, rel, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, rel, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, rel, k))
        metrics["mrr"].append(mrr(retrieved, rel))

    return {m: sum(v) / len(v) for m, v in metrics.items() if v}


class TestE2EEvaluation:
    """Compute retrieval quality metrics across all three backends."""

    @pytest.fixture(scope="module")
    def eval_data(self):
        chunks = _load_json("e2e_chunks.json")
        queries = _load_json("e2e_queries.json")
        relevance = _load_json("e2e_relevance.json")
        return {"chunks": chunks, "queries": queries, "relevance": relevance}

    @pytest.fixture
    def all_backends(self, eval_data):
        """Set up backends. pgvector connects + queries + closes in one loop."""
        chunks = eval_data["chunks"]
        queries = eval_data["queries"]
        relevance = eval_data["relevance"]

        chroma = _chroma_setup(chunks)
        qdrant_path = asyncio.run(_qdrant_setup(chunks))

        # pgvector: do EVERYTHING (connect, setup, search, close) inside one loop
        async def _pg_evaluate():
            pool = await _pg_connect()
            if pool is None:
                return None
            await _pg_setup(pool, chunks)

            results = {}
            for q in queries:
                qid = q["id"]
                rel = set(relevance["judgments"].get(qid, {}).get("relevant", []))
                if not rel:
                    continue
                retrieved = await _pg_search(pool, q["embedding"], 10)
                results[qid] = {"retrieved": retrieved, "relevant": rel}

            await pool.execute("DROP TABLE IF EXISTS e2e_eval")
            await pool.close()
            return results

        pg_results = asyncio.run(_pg_evaluate())

        def _chroma_search_fn(emb, k):
            return _chroma_search(chroma, emb, k)

        def _qdrant_search_fn(emb, k):
            return asyncio.run(_qdrant_search(qdrant_path, emb, k))

        yield {
            "pgvector": pg_results,
            "chromadb": _chroma_search_fn,
            "qdrant": _qdrant_search_fn,
            "queries": queries,
            "relevance": relevance,
        }
        shutil.rmtree(qdrant_path, ignore_errors=True)

    def test_evaluation_report(self, all_backends):
        """Print a comparison report for all backends."""
        print("\n" + "=" * 72)
        print("  RAG Retrieval Quality Evaluation")
        print("  Model: BAAI/bge-small-en-v1.5 (384d) | K=10")
        print("=" * 72)

        # pgvector: results pre-computed inside one event loop
        pg_data = all_backends["pgvector"]
        pg_metrics = _evaluate_precomputed(pg_data) if pg_data else {}

        # ChromaDB / Qdrant: compute on-the-fly
        ch_metrics = _evaluate(
            "chromadb",
            all_backends["chromadb"],
            all_backends["queries"],
            all_backends["relevance"],
        )
        qd_metrics = _evaluate(
            "qdrant",
            all_backends["qdrant"],
            all_backends["queries"],
            all_backends["relevance"],
        )

        results = {"pgvector": pg_metrics, "chromadb": ch_metrics, "qdrant": qd_metrics}

        # Header
        metric_names = ["recall@3", "recall@5", "precision@3", "mrr", "ndcg@5"]
        header = f"{'Backend':<12}" + "".join(f"{m:>12}" for m in metric_names)
        print(header)
        print("-" * len(header))

        for name in ["pgvector", "chromadb", "qdrant"]:
            m = results[name]
            row = f"{name:<12}" + "".join(f"{m.get(k, 0):>12.4f}" for k in metric_names)
            print(row)

        print("-" * len(header))
        print()

        # Per-query detail (uses pgvector pre-computed data)
        print("  Per-query breakdown (pgvector):")
        for q in all_backends["queries"]:
            qid = q["id"]
            data = pg_data.get(qid, {})
            rel = data.get("relevant", set())
            retrieved = data.get("retrieved", [])
            hits = [cid for cid in retrieved if cid in rel]
            label = all_backends["relevance"]["judgments"].get(qid, {}).get("label", qid)
            print(
                f"  {qid} ({label[:30]}): "
                f"recall@5={recall_at_k(retrieved, rel, 5):.2f} "
                f"MRR={mrr(retrieved, rel):.3f} "
                f"hits@10={len(hits)}/{len(rel)}"
            )

        # Assert minimum quality thresholds
        for name in ["pgvector", "chromadb", "qdrant"]:
            m = results[name]
            assert m["recall@5"] >= 0.3, f"{name} recall@5={m['recall@5']:.3f} below threshold 0.3"
            assert m["mrr"] >= 0.4, f"{name} MRR={m['mrr']:.3f} below threshold 0.4"

        print("\n  All backends pass quality thresholds.")
