# Full Configuration Reference

Raglan configuration has three layers: **Builder API** (assemble your pipeline), **Stage config** (per-stage parameters), and **Runtime overrides** (per-request temporary changes).

---

## 1. Builder API

All configuration goes through the `RaglanBuilder`:

```python
rag = (
    Raglan.builder()
    # --- Stage 1: Query Expansion ---
    .with_expander(
        OpenAIExpander(
            model="gpt-4o-mini",
            temperature=0.3,
            prompt_template=None,  # custom prompt template
            base_url=None,  # proxy / compatible endpoint
            api_key=None,  # or set OPENAI_API_KEY env var
        )
    )
    # --- Bridge: Embedding ---
    .with_embedder(
        OpenAIEmbedder(
            model="text-embedding-3-small",
            batch_size=100,
            base_url=None,
            api_key=None,
        )
    )
    # --- Stage 2: Retrievers (parallel) ---
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://user:pass@localhost:5432/db",
                table="kb.chunks",
                id_column="id",
                content_column="content",
                embedding_column="embedding",
                parent_id_column="parent_id",
                metadata_column="attrs",
                distance_metric="cosine",
            ),
            BM25Retriever(k1=1.5, b=0.75),
        ]
    )
    # --- Stage 3: Fusion ---
    .with_fusion(
        RRFFusion(
            k=60,
            dense_weight=0.8,
            sparse_weight=0.2,
            variant_weight=0.3,
        )
    )
    # --- Stage 4: Reranking ---
    .with_reranker(
        CrossEncoderReranker(
            model_name="ms-marco-TinyBERT-L2-v2",
            device=None,  # "cpu" | "cuda" | None (auto)
            batch_size=8,
            min_score=0.0,
        )
    )
    # --- Stage 5: Context Building ---
    .with_context_builder(
        ParentExpander(
            loader=my_parent_loader,
            max_tokens=6000,
        )
    )
    # --- Global ---
    .with_fallback_mode("degrade")  # "degrade" | "strict"
    .with_metrics_collector(LoggingMetricsCollector())
    .build()
)
```

## 2. Stage Configuration Details

### Stage 1: Query Expansion

Expands user query into multiple semantically equivalent variants.

| Expander | Provider | Key params |
|----------|----------|-----------|
| `OpenAIExpander` | OpenAI / compatible API | `model`, `temperature`, `prompt_template`, `base_url`, `api_key` |
| `LiteLLMExpander` | 100+ LLM providers | `model`, `temperature`, `prompt_template`, `api_key`, `api_base` |
| `IdentityExpander` | No-op (default) | Returns original query unchanged |

**Custom prompt template:**

```python
OpenAIExpander(
    prompt_template="""Generate {num_variants} search queries for a legal database.
Original query: {query}
Return JSON only: {{"variants": ["...", "..."]}}""",
)
```

**Skip this stage**: Omit `.with_expander()` — default `IdentityExpander` does nothing.

### Stage 2a: Dense Retrieval

Semantic search using vector embeddings.

| Retriever | Backend | Key params |
|-----------|---------|-----------|
| `ConfigurablePgvectorRetriever` | Postgres + pgvector | `connection_string`, `table`, column mappings, `distance_metric` |
| `QdrantRetriever` | Qdrant (local/cloud) | `url`, `collection_name`, `vector_name`, `distance_metric` |
| `ChromaDBRetriever` | ChromaDB | `collection_name`, `persist_directory`, `distance_metric` |
| `MemoryRetriever` | In-memory (testing) | `chunks` preloaded data |

**Custom retriever**: Implement the `Retriever` protocol and pass to `.with_retrievers()`.

### Stage 2b: Sparse Retrieval (BM25)

| Retriever | Description | Key params |
|-----------|------------|-----------|
| `BM25Retriever` | Pure Python Okapi BM25 | `k1=1.5`, `b=0.75`, custom `tokenizer`, `stopwords` |

**Disable BM25**: Simply don't include it in the retrievers list.

### Stage 3: Fusion

| Fusion | Description | Key params |
|--------|------------|-----------|
| `RRFFusion` | Reciprocal Rank Fusion (default) | `k=60`, `dense_weight`, `sparse_weight`, `variant_weight` |
| `WeightedFusion` | Min-max normalized weighted fusion | `weights` dict per retriever |
| `RoundRobinFusion` | Round-robin interleaving | No params |

### Stage 4: Reranking

| Reranker | Description | Key params |
|----------|------------|-----------|
| `CrossEncoderReranker` | Local sentence-transformers model | `model_name`, `device`, `batch_size`, `min_score` |
| `CohereReranker` | Cohere Rerank API | `model`, `min_score`, `api_key`, `base_url` |

**Skip this stage**: Omit `.with_reranker()`.

**Recommended models:**

| Model | Size | Speed | Accuracy | Use case |
|-------|------|-------|----------|----------|
| `ms-marco-TinyBERT-L2-v2` | 20 MB | Fast | Medium | Default, CPU-friendly |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 80 MB | Medium | Good | Balanced |
| `BAAI/bge-reranker-v2-m3` | 2.2 GB | Slow | High | Multilingual (Chinese + English) |

### Stage 5: Context Building

| Builder | Description | Key params |
|---------|------------|-----------|
| `ParentExpander` | Load full parent chunks | `loader` callable, `max_tokens` |
| `WindowBuilder` | Surround chunk with context window | `loader` callable, `window_chars`, `max_tokens` |
| `PassthroughBuilder` (default) | Return chunks as-is | None |

---

## 3. Runtime Overrides

Per-request options via `SearchOptions`:

```python
from raglan import SearchOptions

results, trace = await rag.search(
    "how to return my order",
    top_k=10,  # override final result count
    options=SearchOptions(
        top_k=10,  # pipeline-level result count
        dense_top_k=30,  # candidates per dense retriever
        bm25_top_k=15,  # candidates per sparse retriever
        reranker_min_score=0.7,  # stricter threshold
        reranker_top_n=5,  # results after reranking
        max_context_tokens=3000,  # less context
        retriever_timeout=5.0,  # per-retriever timeout
        fallback_mode="strict",
    ),
)
```

---

## 4. Middleware

Wrap any stage with cross-cutting behaviour:

| Middleware | Purpose | Key params |
|-----------|---------|-----------|
| `TimeoutMiddleware` | Per-stage timeout | `timeout` seconds |
| `RetryMiddleware` | Retry on transient errors | `max_retries`, `backoff` (exponential/linear/constant) |
| `CircuitBreakerMiddleware` | Skip after repeated failures | `failure_threshold`, `recovery_timeout` |
| `LoggingMiddleware` | Structured stage logging | `logger`, `level` |
| `RateLimiter` | Token-bucket throttling | `rate`, `burst` |

Middleware are placed directly in the Pipeline stages list (not via Builder):

```python
Pipeline(
    [
        TimeoutMiddleware(30.0),
        OpenAIExpander(),
        RetryMiddleware(max_retries=3),
        OpenAIEmbedder(),
        [BM25Retriever(), PgvectorRetriever()],
        RRFFusion(),
        CircuitBreakerMiddleware(5),
        CrossEncoderReranker(),
        PassthroughBuilder(),
    ]
)
```

---

## 5. Resilience

| Utility | Purpose |
|---------|---------|
| `RetryBudget` | Caps total retries in a sliding time window |
| `RateLimiter` | Token-bucket rate limiter, usable as middleware |
| `HealthChecker` | Async health checks for pipeline dependencies |

---

## 6. Observability

| Collector | Purpose |
|-----------|---------|
| `NoOpMetricsCollector` | Default, does nothing |
| `LoggingMetricsCollector` | Logs per-stage timing via Python logging |

Custom: implement the `MetricsCollector` protocol for Prometheus, Datadog, etc.

```python
rag = (
    Raglan.builder()
    .with_retrievers([...])
    .with_metrics_collector(LoggingMetricsCollector())
    .build()
)
```

---

## 7. Serialization

```python
# Export
config = rag.export_config()

# Import
rag = Raglan.from_dict(config)
```

---

## 8. Configuration Scenarios

### Speed first (< 200ms)

```python
rag = (
    Raglan.builder()
    .with_retrievers([BM25Retriever()])  # BM25 only, no embedding API calls
    .build()
)
```

### Recall first

```python
rag = (
    Raglan.builder()
    .with_expander(OpenAIExpander(model="gpt-4o-mini"))
    .with_embedder(OpenAIEmbedder())
    .with_retrievers([PgvectorRetriever(...), BM25Retriever()])
    .with_fusion(RRFFusion(dense_weight=0.7, sparse_weight=0.3))
    .with_reranker(
        CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            min_score=0.3,
        )
    )
    .build()
)
```

### CPU-only

```python
rag = (
    Raglan.builder()
    .with_embedder(
        HuggingFaceEmbedder(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        )
    )
    .with_retrievers([BM25Retriever(), MemoryRetriever()])
    .with_reranker(
        CrossEncoderReranker(
            model_name="ms-marco-TinyBERT-L2-v2",
            device="cpu",
            batch_size=1,
        )
    )
    .build()
)
```

### Pure keyword search (no vectors)

```python
rag = Raglan.builder().with_retrievers([BM25Retriever()]).build()
```

### Serialized / reloaded pipeline

```python
# First run
rag = Raglan.builder().with_retrievers([BM25Retriever()]).build()
config = rag.export_config()
save_to_file(config, "pipeline.json")

# Later
config = load_from_file("pipeline.json")
rag = Raglan.from_dict(config)
```
