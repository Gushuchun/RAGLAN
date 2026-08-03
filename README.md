# Raglan

<p align="center">
  <a href="https://pypi.org/project/raglan-retrieval/"><img src="https://img.shields.io/pypi/v/raglan.svg" alt="PyPI"></a>
  <a href="https://github.com/Gushuchun/RAGLAN/actions"><img src="https://img.shields.io/github/actions/workflow/status/Gushuchun/RAGLAN/ci.yml?branch=master" alt="CI"></a>
  <a href="https://github.com/Gushuchun/RAGLAN/actions"><img src="https://img.shields.io/badge/coverage-≥85%25-green" alt="Coverage"></a>
  <a href="https://github.com/Gushuchun/RAGLAN/blob/master/LICENSE"><img src="https://img.shields.io/github/license/Gushuchun/RAGLAN" alt="License"></a>
  <a href="https://pypi.org/project/raglan-retrieval/"><img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python"></a>
</p>

A lightweight, highly configurable RAG retrieval engine. Framework-free, protocol-driven.

## Why Raglan?

Standard RAG has one step: vector search → top-K results. Real-world retrieval needs more:

| Problem | Standard RAG | Raglan |
|---------|-------------|--------|
| Single query misses aspects | "return policy" finds policy text, misses refund flow | Auto-generates 3 query variants, searches in parallel |
| Dense-only retrieval | Poor short-keyword / jargon matching | BM25 sparse + vector dense, RRF hybrid fusion |
| Small chunks lack context | "refund takes 7 days" without surrounding doc | Child-chunk match → expanded to full parent context |
| Vector similarity != semantic match | "how to return" vs "I don't want this" look close | Cross-Encoder pair-wise reranking filters false positives |
| One failure kills the pipeline | Embedding service timeout → empty results | Each stage degrades independently, rest continues |

## Six-Stage Pipeline

```
User query
  │
  ▼
┌─────────────────────────────────┐
│ Stage 1: QueryExpander           │  ← LLM generates entity extraction + 3 variants
└──────────────┬──────────────────┘
               │ original + variants = parallel search
               ▼
┌─────────────────────────────────┐
│ Stage 2: Retrievers (parallel)   │
│  · Dense: pgvector / Qdrant / ChromaDB / ...
│  · Sparse: BM25 full-text        │
└──────────────┬──────────────────┘
               │ multi-source results
               ▼
┌─────────────────────────────────┐
│ Stage 3: Fusion                  │  ← RRF / Weighted / RoundRobin
└──────────────┬──────────────────┘
               │ fused candidates
               ▼
┌─────────────────────────────────┐
│ Stage 4: Reranker (optional)     │  ← Cross-Encoder / Cohere Rerank
│ Filters scores below threshold   │
└──────────────┬──────────────────┘
               │ top-N candidates
               ▼
┌─────────────────────────────────┐
│ Stage 5: ContextBuilder          │  ← Parent expansion / window / passthrough
│ Greedy fill up to max_tokens     │
└──────────────┬──────────────────┘
               │
               ▼
         Final Top-N results
```

## Installation

```bash
pip install raglan-retrieval
```

With optional providers:

```bash
pip install raglan-retrieval[openai]        # OpenAI embedder + expander
pip install raglan-retrieval[pgvector]      # Postgres + pgvector retriever
pip install raglan-retrieval[huggingface]   # HF embedder + Cross-Encoder reranker
pip install raglan-retrieval[qdrant]        # Qdrant retriever
pip install raglan-retrieval[chromadb]      # ChromaDB retriever
pip install raglan-retrieval[cohere]        # Cohere reranker
pip install raglan-retrieval[dashscope]     # Alibaba DashScope embedder
pip install raglan-retrieval[litellm]       # LiteLLM multi-provider expander
pip install raglan-retrieval[all]           # Everything
```

## 5-Minute Quickstart

```python
from raglan import Raglan
from raglan.retrievers import BM25Retriever
import asyncio


async def main():
    # Step 1: Set up a retriever with your data
    bm25 = BM25Retriever()

    async def chunks():
        yield [
            ("doc1", "Return policy: items can be returned within 30 days.", None),
            ("doc2", "Refund process: refunds are issued to the original payment method.", None),
            ("doc3", "Shipping: orders ship within 2 business days.", None),
        ]

    await bm25.index(chunks())

    # Step 2: Build the pipeline (all stages optional beyond retrievers)
    rag = Raglan([bm25])

    # Step 3: Search
    results, trace = await rag.search("how to return my order")

    for r in results:
        print(f"[{r.score:.3f}] {r.content}")

    print(f"Pipeline took {trace.total_ms:.1f}ms")


asyncio.run(main())
```

### Incremental configuration

Prefer building up a pipeline piece by piece? Start with an empty
instance and add stages incrementally — no `build()` needed:

```python
from raglan import Raglan
from raglan.retrievers import BM25Retriever

rag = Raglan()  # empty, configurable
rag.add_retriever(bm25)  # add a retriever
rag.set_embedder("openai:text-embedding-3-small")  # string shorthand
rag.set_expander("openai:gpt-4o-mini")  # vendor:model form
rag.set_fusion("rrf")
# every setter returns self — chainable:
#   Raglan().add_retriever(bm25).set_fusion("rrf")

results, trace = await rag.search("my query")
```

Components accept three forms: a live object (`BM25Retriever()`), a
`"vendor:model"` string (`"openai:text-embedding-3-small"`, `"rrf"`), or a
`{"type": ..., "params": ...}` dict — the same format used by `from_dict()`.

### Configuration templates

For team-shared or file-based configuration, grab a template and fill it in:

```python
cfg = Raglan.config()  # → {"retrievers": [], "fusion": "rrf", "expander": None, ...}
cfg["retrievers"].append(
    {"type": "pgvector", "params": {"connection_string": "...", "table": "kb.chunks"}}
)
cfg["expander"] = "openai:gpt-4o-mini"

rag = Raglan.from_config(cfg)
```

The template is JSON/YAML-serialisable, so configs can live in a file:

```yaml
# raglan.yaml
retrievers:
  - type: pgvector
    params: {connection_string: "postgresql://...", table: kb.chunks}
  - type: bm25
fusion: rrf
expander: "openai:gpt-4o-mini"
```

### Adding vector search and reranking

```python
from raglan.context_builders import ParentExpander
from raglan.embedders import OpenAIEmbedder
from raglan.expanders import OpenAIExpander
from raglan.rerankers import CrossEncoderReranker
from raglan.retrievers import ConfigurablePgvectorRetriever

rag = (
    Raglan.builder()
    .with_expander(OpenAIExpander(model="gpt-4o-mini"))
    .with_embedder(OpenAIEmbedder(model="text-embedding-3-small"))
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://...",
                table="kb.chunks",
                embedding_column="embedding",
                parent_id_column="parent_id",
            ),
            BM25Retriever(),
        ]
    )
    .with_reranker(
        CrossEncoderReranker(
            model_name="ms-marco-TinyBERT-L2-v2",
            min_score=0.5,
        )
    )
    .with_context_builder(
        ParentExpander(
            loader=my_parent_loader,
            max_tokens=6000,
        )
    )
    .with_fallback_mode("degrade")
    .build()
)

results, trace = await rag.search("damaged order return policy")

# Pass request-scoped context (identity, tenant, permissions) down to retrievers
results, trace = await rag.search(
    "damaged order return policy",
    request={"user_id": "u123", "tenant": "acme"},
)
```

## Design Philosophy

1. **Framework-Free** — No dependency on LangChain, LlamaIndex, or any specific vector database or LLM provider. Protocols define the interfaces; you provide the implementations.

2. **Graceful Degradation** — Every stage is independent. Query expansion fails? Use the original query. Cross-Encoder not installed? Skip reranking. BM25 unavailable? Pure vector search still works.

3. **Fully Configurable** — Every stage's parameters, weights, models, and thresholds are configurable via the Builder or `from_dict()`.

4. **Production-Ready** — Extracted from production systems processing thousands of support emails daily. Handles extreme text, multilingual, and high-concurrency scenarios.

## Comparison with Existing RAG Frameworks

| Tool | Positioning | vs Raglan |
|------|------------|-----------|
| **LangChain RAG** | Full-stack LLM framework's RAG module | LC binds to LangChain ecosystem; Raglan is zero-dependency, direct use |
| **LlamaIndex** | Data→LLM full pipeline | LI has many concepts (Node, Index, QueryEngine); Raglan has one: Search |
| **RAGatouille** | ColBERT-specific RAG | RAGatouille focuses on ColBERT; Raglan is general retrieval + reranking |
| **Cohere Rerank** | Commercial API reranking | Cohere charges, sends data to cloud; Raglan runs locally |
| **rerankers (answer.ai)** | Unified reranking API | rerankers only does reranking; Raglan covers retrieval→fusion→reranking |

## Project Structure

```
raglan/
├── README.md
├── docs/
│   ├── architecture.md      # Architecture design
│   ├── configuration.md     # Full configuration reference
│   ├── pipeline.md          # Six-stage deep dive
│   └── examples.md          # Multi-scenario examples
├── raglan/
│   ├── __init__.py
│   ├── raglan.py            # Raglan facade + Builder
│   ├── pipeline.py          # Pipeline engine + stage dispatch
│   ├── protocols.py         # User-implementable abstract interfaces
│   ├── types.py             # ScoredChunk, SearchResult, Filter, etc.
│   ├── exceptions.py        # Exception hierarchy
│   ├── observability.py     # Metrics collector
│   ├── expanders/           # Query expansion (OpenAI, LiteLLM, Identity)
│   ├── embedders/           # Embedding (OpenAI, HuggingFace, DashScope)
│   ├── retrievers/          # Search backends (BM25, pgvector, Qdrant, ChromaDB)
│   ├── fusion/              # Result fusion (RRF, Weighted, RoundRobin)
│   ├── rerankers/           # Reranking (CrossEncoder, Cohere)
│   ├── context_builders/    # Context assembly (Parent, Window, Passthrough)
│   ├── middleware/           # Timeout, Retry, CircuitBreaker, Logging
│   └── resilience/          # RateLimiter, RetryBudget, HealthChecker
└── tests/
```

## License

MIT
