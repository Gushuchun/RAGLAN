# Changelog

All notable changes to Raglan will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] — 2026-08-03

### Added

- **Request context**: `search(request={...})` threads per-request context
  (identity, tenant, permissions) through to `Retriever.retrieve(request=...)`.
- **`where_builder`** callback on `ConfigurablePgvectorRetriever` for
  parameterised ACL/visibility WHERE predicates.
- **`client_factory`** on `OpenAIEmbedder`/`OpenAIExpander` to route through a
  gateway or custom client proxy.

## [0.2.1] — 2026-08-03

### Fixed

- `RRFFusion` and `BM25Retriever` now preserve chunk metadata end-to-end.
- Consecutive middleware on one stage now work correctly (and cleanup reaches
  the wrapped stage); an orphan middleware raises a clear error.
- SQLAlchemy-backed pgvector retrieval binds query parameters correctly.

### Added

- Pluggable sparse index: `BM25Retriever(index=...)` accepts an external
  backend (Elasticsearch, meilisearch, ...).
- `register_component()` for config-driven construction of custom stages.
- `ConfigurablePgvectorRetriever(session_factory=...)` for SQLAlchemy apps.
- `warm_up()` lifecycle for pre-loading models at startup.
- `Trace` now exposes intermediate results (`expanded_queries`, `entities`,
  per-retriever hit counts); `search(trace_level=...)` overrides per request.

### Performance

- BM25 concurrent searches no longer serialise on the writer lock; average
  document length no longer drifts under incremental add/remove.

## [0.2.0] — 2026-07-31

### Added

- **Direct instantiation**: `Raglan([bm25])` — pass a retriever (or list) as the
  first positional argument; all other stages optional with defaults.
- **Incremental configuration**: `Raglan()` empty instance + `add_retriever()`,
  `set_embedder()`, `set_expander()`, `set_fusion()`, `set_reranker()`,
  `set_context_builder()`, `set_fallback_mode()`, `set_trace_level()` — all
  return `self` for chaining; the pipeline is assembled lazily on first search.
- **Configuration templates**: `Raglan.config()` returns a defaults-filled dict
  (JSON/YAML-serialisable); `Raglan.from_config()` builds from it.
- **Flexible component forms**: every stage accepts a live object, a
  `"vendor:model"` string shorthand (`"openai:text-embedding-3-small"`,
  `"rrf"`), or a `{"type": ..., "params": ...}` dict — in constructors,
  setters, and `from_dict()` alike.

### Changed

- `Raglan.__init__` accepts retrievers positionally while remaining backward
  compatible with the legacy `Raglan(pipeline, config)` constructor.
- `RaglanBuilder.add_retriever()` appends to the retriever list (the existing
  `with_retrievers()` still replaces).
- `Raglan.export_config()` now serialises the current builder state for
  unbuilt incremental instances.

## [0.1.0] — 2026-07-29

### Added

**Pipeline Engine**

- Six-stage composable pipeline: QueryExpander → Embedder → Retrievers → Fusion → Reranker → ContextBuilder.
- Parallel retriever execution with automatic result deduplication by name.
- Graceful degradation per stage (configurable: `degrade` or `strict`).
- Global timeout with partial result preservation.
- `TraceLevel` filtering: `minimal` (timings only), `normal` (+ degradations), `full` (+ raw data).
- Protocol-driven dispatch table — users add custom stages without modifying the engine.

**Protocols**

- `QueryExpander`, `Embedder`, `Retriever`, `IndexableRetriever`, `Fusion`, `Reranker`, `ContextBuilder` — all `@runtime_checkable`, no base class required.
- `Middleware` protocol for cross-cutting stage wrappers.
- `MetricsCollector` protocol for observability integration.

**Retrievers**

- `BM25Retriever`: pure Python Okapi BM25 with CJK bigram tokenizer, async-safe double-buffered index rebuild, `heapq.nlargest` optimised retrieval, configurable stopwords and tokenizer.
- `ConfigurablePgvectorRetriever`: column-name mapping adapter for PostgreSQL + pgvector, automatic SQL generation with validated identifiers, JSONB metadata filter translation, loop-change detection for `search_sync()` safety.
- `ChromaDBRetriever`: in-memory, persistent, and client-server (`HttpClient`) modes, distance-to-score conversion for cosine/L2/IP, metadata filter translation.
- `QdrantRetriever`: in-memory and HTTP/gRPC server modes, deterministic UUID-based point IDs, recursive AND/OR filter support, vector-size aware collection creation.
- `MemoryRetriever`: brute-force cosine similarity for testing and small datasets, `asyncio.Lock` protected.

**Query Expanders**

- `OpenAIExpander`: query variant generation via OpenAI chat completion, configurable prompt template.
- `LiteLLMExpander`: 100+ LLM provider support via LiteLLM, graceful fallback for providers without `response_format`.
- `IdentityExpander`: no-op default.

**Embedders**

- `OpenAIEmbedder`: OpenAI Embeddings API with configurable batch size and dimension detection.
- `HuggingFaceEmbedder`: local sentence-transformers models with eager dimension loading, `asyncio.Lock` serialisation for thread safety.
- `DashScopeEmbedder`: Alibaba DashScope (Tongyi) API with per-request API key (no global state mutation).

**Fusion Strategies**

- `RRFFusion`: Reciprocal Rank Fusion with per-source-type weight normalisation (dense/sparse).
- `WeightedFusion`: min-max normalized weighted score fusion with metadata preservation.
- `RoundRobinFusion`: round-robin interleaving with parent-chunk deduplication.

**Rerankers**

- `CrossEncoderReranker`: local sentence-transformers Cross-Encoder with configurable `input_builder`.
- `CohereReranker`: Cohere Rerank API v2 integration.

**Context Builders**

- `ParentExpander`: parent-chunk context loading with `tiktoken`-based (or CJK-aware heuristic) token counting and greedy budget packing.
- `WindowBuilder`: sliding window context extraction around matched chunks.
- `PassthroughBuilder`: no-op default.

**Filter System**

- Operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `exists`, `contains`.
- Boolean combinators: `Filter.all(...)` / `Filter.any(...)` with `&` / `|` operator overloads.
- Backend-specific translation for pgvector (SQL WHERE clause), ChromaDB (`$and`/`$or` dict), Qdrant (`must`/`should`/`must_not`).
- Unsupported operators raise `FilterError` rather than silently ignoring.

**Middleware**

- `TimeoutMiddleware`: per-stage deadline with configurable timeout.
- `RetryMiddleware`: exponential/linear/constant backoff, configurable retryable exception types.
- `CircuitBreakerMiddleware`: open/closed/half-open state machine, lock released during stage execution.
- `LoggingMiddleware`: entry/exit/timing/exception logging.
- `RateLimiter`: token-bucket throttle, re-exported from `raglan.middleware`.

**Resilience**

- `RetryBudget`: sliding-window retry cap with async-safe `available_async()`.
- `RateLimiter`: token-bucket algorithm with `acquire()` and `wait_and_acquire()`.
- `HealthChecker`: parallel async health checks with per-check timeouts and register/unregister.

**Observability**

- `MetricsCollector` protocol: `record_search()` and `record_stage()` hooks.
- `NoOpMetricsCollector`: default no-op.
- `LoggingMetricsCollector`: production-safe by default — queries redacted unless `log_queries=True`.

**Configuration & Serialization**

- `Raglan.builder()` fluent API with defaulted stages.
- `from_dict()` / `export_config()` round-trip serialization with component registry.
- `SearchOptions` per-request overrides for all stage parameters.
- `TraceLevel` configurable via `RaglanBuilder.with_trace_level()`.

**Resource Management**

- `async with Raglan(...)` context manager support.
- `Raglan.close()` iterates all stages via public `Pipeline.iter_stages()`.
- `close()` on all stateful components: pgvector pool, Qdrant client, ChromaDB client, OpenAI clients, Cohere client.
- Lazy pool/client initialization protected by `asyncio.Lock` with double-check.

**Testing & Quality**

- 355 tests across unit, integration, e2e, benchmark, property, and regression categories.
- Parallel execution via `pytest-xdist -n auto`.
- Real-backend integration tests: PostgreSQL + pgvector, ChromaDB (in-memory + Docker HTTP), Qdrant (in-memory + Docker HTTP).
- Pre-computed test fixtures: 127-chunk RAG article dataset with BGE-small embeddings, 7 queries with relevance judgments.
- Benchmark suite: multi-config comparison (BM25 / Dense / Hybrid) with recall, precision, MRR, NDCG, and P50/P95 latency.
- Hypothesis property-based tests for filter system, BM25 tokenizer, RRF invariants, and chunk parser.
- Memory stability tests with RSS monitoring.

**CI/CD**

- GitHub Actions: lint (ruff), typecheck (mypy strict), test matrix (Python 3.10–3.14), integration tests (pgvector/ChromaDB/Qdrant Docker services), security scan (Bandit + pip-audit), README validation.
- `pre-commit` hooks: ruff check, ruff format, mypy, pytest.

**Documentation**

- `docs/architecture.md`: design principles, concurrency model, data flow, extension points.
- `docs/configuration.md`: full Builder API reference, middleware table, scenario presets.
- `docs/pipeline.md`: six-stage deep dive with formulas and failure modes.
- `docs/examples.md`: 12 usage examples covering multiple backends and deployment scenarios.
- `benchmark/README.md`, `tests/README.md`, `tests/regression/README.md`.
- Issue templates: bug report and feature request.
- Complete docstrings on all public classes and Builder methods.

**Dependencies**

- Zero mandatory dependencies beyond Python stdlib.
- All integrations are optional extras: `[openai]`, `[pgvector]`, `[chromadb]`, `[qdrant]`, `[cohere]`, `[huggingface]`, `[dashscope]`, `[litellm]`, `[tiktoken]`, `[all]`.
- Consistent lazy-import with user-friendly error messages via `_lazy._import_module()`.

[0.2.2]: https://github.com/Gushuchun/RAGLAN/releases/tag/v0.2.2
[0.2.1]: https://github.com/Gushuchun/RAGLAN/releases/tag/v0.2.1
[0.2.0]: https://github.com/Gushuchun/RAGLAN/releases/tag/v0.2.0
[0.1.0]: https://github.com/Gushuchun/RAGLAN/releases/tag/v0.1.0
