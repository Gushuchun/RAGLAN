# Architecture Design

## Overview

Raglan uses a **Protocol-Driven Architecture**. The core engine depends only on Python Protocol definitions — never on concrete implementations. All external dependencies (vector databases, LLM SDKs, embedding services) are injected by the user.

```
┌──────────────────────────────────────────────────────┐
│                     Raglan (facade)                    │
│  ┌─────────────────────────────────────────────────┐  │
│  │                  Pipeline Engine                  │  │
│  │                                                  │  │
│  │  ┌──────────┐  ┌─────────┐  ┌───────────────┐   │  │
│  │  │ Expander │→ │Embedder │→ │  Retrievers   │   │  │
│  │  │ (Stage 1)│  │ (bridge)│  │  (Stage 2)    │   │  │
│  │  └──────────┘  └─────────┘  └───────┬───────┘   │  │
│  │                                      │            │  │
│  │  ┌──────────────┐  ┌──────────┐     │            │  │
│  │  │ContextBuilder│← │ Reranker │←────┘            │  │
│  │  │  (Stage 5)   │  │(Stage 4) │  ┌──────────┐   │  │
│  │  └──────────────┘  └──────────┘  │  Fusion  │   │  │
│  │                                   │ (Stage 3)│   │  │
│  │                                   └──────────┘   │  │
│  │                                                  │  │
│  │  Middleware: Timeout | Retry | CircuitBreaker    │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
         │            │            │
         ▼            ▼            ▼
┌──────────────────────────────────────────────────────┐
│              Protocol Layer (user implements)          │
│                                                       │
│  QueryExpander   Embedder   Retriever                 │
│  Fusion          Reranker   ContextBuilder            │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Dependency Inversion

The core engine never `import`s any external database, LLM SDK, or vector store library. It depends only on Protocol signatures. Users implement their PostgreSQL, Milvus, Qdrant, or OpenAI API logic and inject it.

```
Traditional (tight coupling):
  RAG Engine → import pgvector → import OpenAI SDK → import SQLAlchemy

Raglan (dependency inversion):
  RAG Engine → Protocol ← User implementation ← pgvector / OpenAI / SQLAlchemy
```

### 2. Every Stage is Independently Replaceable

Any of the pipeline stages can be:
- **Skipped**: Omit from Builder (defaults to no-op: `IdentityExpander`, `PassthroughBuilder`)
- **Replaced**: Provide a different implementation (e.g., swap BM25 for Elasticsearch)
- **Extended**: Wrap with middleware (timeout, retry, circuit breaker)

Custom components can be registered for config-driven construction:

```python
from raglan import register_component

register_component("my_retriever", MyRetriever)
rag = Raglan.from_dict({"retrievers": [{"type": "my_retriever", "params": {...}}]})
```

Stages may also expose an optional `async def warm_up()` for eager
initialisation (e.g. pre-loading a Cross-Encoder model). Call it through the
facade during startup:

```python
await rag.warm_up()  # pre-loads any stage that supports it
```

```python
# Minimal: BM25 only — direct instantiation
rag = Raglan([BM25Retriever()])

# Incremental — configure piece by piece, no build()
rag = Raglan()
rag.add_retriever(BM25Retriever())
rag.set_embedder("openai:text-embedding-3-small")

# Full: Query expansion + embedding + dual retrieval + RRF + reranking
# (Builder style for explicit, typed configuration)
rag = (
    Raglan.builder()
    .with_expander(OpenAIExpander(model="gpt-4o-mini"))
    .with_embedder(OpenAIEmbedder())
    .with_retrievers([PgvectorRetriever(...), BM25Retriever()])
    .with_fusion(RRFFusion(k=60))
    .with_reranker(CrossEncoderReranker(min_score=0.5))
    .with_context_builder(ParentExpander(loader=my_loader))
    .build()
)
```

### 3. Graceful Degradation

Each stage failure does not affect other stages. The degradation chain:

```
QueryExpander fails  → Use original query only
Dense retriever fails → Use sparse results only (and vice versa)
Fusion fails         → Return raw retriever results
Reranker fails       → Skip reranking, keep fused Top-K
ContextBuilder fails → Return child-chunk text as-is
Global timeout       → Return partial results with degradation records
```

Failure information is recorded in the `Trace` object returned with every search.

### 4. Middleware Pipeline

Cross-cutting concerns are handled by middleware that wraps individual stages:

```python
Pipeline(
    [
        TimeoutMiddleware(30.0),
        OpenAIExpander(),  # Query expansion with 30s timeout
        RetryMiddleware(max_retries=3),
        OpenAIEmbedder(),  # Embedding with retry
        [BM25Retriever(), PgvectorRetriever()],  # Parallel retrieval
        RRFFusion(),
        CircuitBreakerMiddleware(failure_threshold=5),
        CrossEncoderReranker(),
    ]
)
```

## Data Flow

```
Input: natural language query string
  │
  ▼
PipelineContext (mutable state bag flowing through stages)
  │
  ├── .query                   # Original user query
  ├── .expanded_queries        # Stage 1: query + variants
  ├── .embeddings              # Bridge: dense vectors
  ├── .retriever_results       # Stage 2: per-retriever ranked lists
  ├── .fused_candidates        # Stage 3: merged + deduplicated
  ├── .reranked_candidates     # Stage 4: cross-encoder rescored
  ├── .final_results           # Stage 5: SearchResult objects
  ├── .degradations            # Accumulated stage failure records
  └── .stage_timings           # Per-stage wall-clock timings
  │
  ▼
Output: (list[SearchResult], Trace)
  ├── Trace.total_ms           # End-to-end latency
  ├── Trace.stage_timings      # Per-stage timings
  ├── Trace.expanded_queries   # Stage 1 variants (non-minimal level)
  ├── Trace.entities           # Extracted entities (non-minimal level)
  ├── Trace.retriever_hits     # Per-retriever hit counts (non-minimal)
  └── Trace.degradations       # Skipped/failed stages
```

> **Trace level filtering.** `Trace` carries intermediate results
> (`expanded_queries`, `entities`, `retriever_hits`) only when
> `trace_level != "minimal"`. Configure it at construction time or override
> per request via `Raglan.search(..., trace_level=...)`.

```python
@dataclass
class ScoredChunk:
    """Internal pipeline currency — flows between stages 2-4."""

    chunk_id: str
    content: str
    score: float
    parent_chunk_id: str | None = None
    chunk_metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class SearchResult:
    """Final output — returned to the caller."""

    chunk_id: str
    content: str
    score: float
    parent_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
```

## Concurrency Model

All I/O operations (vector search, LLM calls, embedding generation) are async. CPU-intensive work (BM25, Cross-Encoder) runs via `asyncio.to_thread` to avoid blocking the event loop.

```
main coroutine
  ├── QueryExpander.expand()              # I/O: LLM API call
  ├── Embedder.embed()                    # I/O: embedding API / to_thread(HF)
  ├── asyncio.gather(
  │     Retriever.retrieve() × N          # I/O: parallel vector searches
  │   )
  ├── Fusion.fuse()                       # CPU: lightweight scoring
  ├── Reranker.rerank()                   # CPU: to_thread(cross-encoder)
  └── ContextBuilder.build()              # I/O: parent document loading
```

## Extension Points

| Extension Point | Protocol | Built-in Implementations |
|----------------|----------|------------------------|
| Query Expansion | `QueryExpander` | `OpenAIExpander`, `LiteLLMExpander`, `IdentityExpander` |
| Embedding | `Embedder` | `OpenAIEmbedder`, `HuggingFaceEmbedder`, `DashScopeEmbedder` |
| Dense Retrieval | `Retriever` | `ConfigurablePgvectorRetriever`, `QdrantRetriever`, `ChromaDBRetriever`, `MemoryRetriever` |
| Sparse Retrieval | `Retriever` | `BM25Retriever` |
| Fusion | `Fusion` | `RRFFusion`, `WeightedFusion`, `RoundRobinFusion` |
| Reranking | `Reranker` | `CrossEncoderReranker`, `CohereReranker` |
| Context Building | `ContextBuilder` | `ParentExpander`, `WindowBuilder`, `PassthroughBuilder` |
| Observability | `MetricsCollector` | `NoOpMetricsCollector` |

## Configuration Serialization

Pipelines can be serialized to/from dictionaries for storage in config files or databases:

```python
# Export
config = rag.export_config()
# {"expander": {"type": "openai_expander", "params": {"model": "gpt-4o-mini", ...}},
#  "embedder": {"type": "openai_embedder", "params": {...}},
#  "retrievers": [{"type": "bm25", "params": {...}}, ...],
#  ...}

# Import
rag = Raglan.from_dict(config)
```
