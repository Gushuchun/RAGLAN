#!/usr/bin/env python3
"""Generate pre-computed test fixtures for the e2e pipeline tests.

Run once (requires network for model download)::

    HF_ENDPOINT=https://hf-mirror.com python tests/generate_e2e_fixtures.py

Output (written to ``tests/fixtures/``):

* ``e2e_chunks.json``   —  parent-child chunks with pre-computed 384-dim embeddings
* ``e2e_queries.json``  —  test queries with pre-computed embeddings + variants

After generation, the e2e tests run without any model or network dependency.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Chunk parser (deterministic — same output for same input)
# ---------------------------------------------------------------------------


def parse_document(text: str) -> list[dict]:
    """Parse markdown into a parent-child chunk tree.

    Returns a list of ``{chunk_id, content, parent_id, metadata}`` dicts.
    """
    # Split on markdown headings
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks: list[dict] = []
    parent_counter = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"(#{1,3})\s+(.+)", section)
        heading = heading_match.group(2) if heading_match else "Frontmatter"

        parent_id = f"parent_{parent_counter:03d}"
        parent_counter += 1

        body_start = heading_match.end() + 1 if heading_match else 0
        body = section[body_start:].strip()

        parent_content = f"## {heading}\n\n{body[:500]}"
        chunks.append(
            {
                "chunk_id": parent_id,
                "content": parent_content,
                "parent_id": "",
                "metadata": {"type": "parent", "heading": heading},
            }
        )

        paragraphs = re.split(r"\n\n+", body)
        child_idx = 0
        for para in paragraphs:
            para = para.strip()
            if len(para) < 40 and not child_idx:
                continue
            if len(para) < 30:
                continue

            child_id = f"{parent_id}_c{child_idx:02d}"
            child_idx += 1
            chunks.append(
                {
                    "chunk_id": child_id,
                    "content": para,
                    "parent_id": parent_id,
                    "metadata": {"type": "child", "heading": heading},
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], model) -> list[list[float]]:
    """Batch-embed and return as plain float lists."""
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [e.tolist() for e in embs]


# ---------------------------------------------------------------------------
# 3. Query definitions (hand-written, covers key topics)
# ---------------------------------------------------------------------------


TEST_QUERIES = [
    {
        "id": "q001",
        "query": "What are the different chunking strategies?",
        "label": "chunking strategies",
        "variants": [
            "How should documents be split into chunks?",
            "What chunking methods exist for RAG?",
            "Explain fixed size vs semantic chunking",
        ],
    },
    {
        "id": "q002",
        "query": "Which embedding models are popular?",
        "label": "embedding models",
        "variants": [
            "What are the best embedding models?",
            "List popular text embedding models",
            "Compare BGE, OpenAI, and other embedding models",
        ],
    },
    {
        "id": "q003",
        "query": "How does hybrid retrieval combine dense and sparse search?",
        "label": "hybrid retrieval",
        "variants": [
            "What is hybrid search in RAG?",
            "How to combine vector search with keyword search?",
            "Explain dense vs sparse retrieval combination",
        ],
    },
    {
        "id": "q004",
        "query": "Why is reranking needed after retrieval?",
        "label": "reranking purpose",
        "variants": [
            "What is the purpose of a reranker?",
            "How does reranking improve search quality?",
            "Why add a second ranking stage?",
        ],
    },
    {
        "id": "q005",
        "query": "How to fix hallucination in RAG systems?",
        "label": "hallucination solutions",
        "variants": [
            "What causes hallucinations in RAG?",
            "How to prevent AI from making up information?",
            "Solutions for LLM hallucination with retrieval",
        ],
    },
    {
        "id": "q006",
        "query": "What is a vector database and how does it work?",
        "label": "vector databases",
        "variants": [
            "How do vector databases store embeddings?",
            "Explain pgvector and Qdrant",
            "What databases support semantic search?",
        ],
    },
    {
        "id": "q007",
        "query": "How to evaluate RAG system quality?",
        "label": "RAG evaluation",
        "variants": [
            "What metrics measure RAG performance?",
            "How to assess retrieval accuracy?",
            "What is recall and precision in RAG?",
        ],
    },
]


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


def main():
    doc_path = Path(__file__).parent / "fixtures" / "rag_article.md"
    if not doc_path.exists():
        print(f"ERROR: {doc_path} not found")
        sys.exit(1)

    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    # Parse
    text = doc_path.read_text(encoding="utf-8")
    chunks = parse_document(text)
    print(
        f"Parsed {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c['metadata']['type'] == 'parent')} parents, "
        f"{sum(1 for c in chunks if c['metadata']['type'] == 'child')} children)"
    )

    # Embed
    print("Loading embedding model (BAAI/bge-small-en-v1.5) ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"Model loaded: {model.get_embedding_dimension()} dimensions")

    chunk_texts = [c["content"] for c in chunks]
    chunk_embs = embed_texts(chunk_texts, model)

    for chunk, emb in zip(chunks, chunk_embs, strict=True):
        chunk["embedding"] = emb

    # Embed queries + variants
    for q in TEST_QUERIES:
        q["embedding"] = embed_texts([q["query"]], model)[0]
        q["variant_embeddings"] = [embed_texts([v], model)[0] for v in q["variants"]]

    # Save
    chunks_path = fixtures_dir / "e2e_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(
        f"Saved {len(chunks)} chunks → {chunks_path} ({chunks_path.stat().st_size / 1024:.0f} KB)"
    )

    queries_path = fixtures_dir / "e2e_queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(TEST_QUERIES, f, ensure_ascii=False)
    print(
        f"Saved {len(TEST_QUERIES)} queries → {queries_path} "
        f"({queries_path.stat().st_size / 1024:.0f} KB)"
    )

    print(
        "\nDone. Tests can now run without model/network:"
        "\n  pytest tests/test_e2e_precomputed.py -v -s"
    )


if __name__ == "__main__":
    main()
