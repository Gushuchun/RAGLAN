# Six-Stage Pipeline Deep Dive

Each stage's input, output, internal implementation, and design rationale.

---

## Stage 1: Query Expansion (`QueryExpander`)

**Goal**: Expand a user query into multiple semantically equivalent variants, covering the same need from different angles.

**Input**:
```
User query: "my order arrived damaged, how do I return it?"
```

**LLM processing**: Built-in prompt does two things — extract key entities + generate query variants:

```
Variants:
  1. "damaged product return process"
  2. "received damaged item refund request"
  3. "broken item exchange policy"
```

**Output**:
```python
[
    "my order arrived damaged, how do I return it?",  # original first
    "damaged product return process",
    "received damaged item refund request",
    "broken item exchange policy",
]
```

**Why it works**: Users write "how to return", but the knowledge base may contain "refund process", "exchange policy", "after-sales handling", "RMA application". Multi-angle queries cover the synonym space.

**Built-in implementations**: `OpenAIExpander`, `LiteLLMExpander`, `IdentityExpander` (no-op default).

**On failure**: LLM call fails, times out, or returns bad format → pipeline continues with original query only. Degradation recorded in `Trace`.

---

## Stage 2a: Dense Retrieval (`Retriever` with `requires_embeddings=True`)

**Goal**: Semantic search using vector embeddings.

**Flow**:
```
4 query embeddings (original + 3 variants)
    │
    ├── embed(query_0)  →  pgvector.search(vec_0, top_k=20)  → 20 results
    ├── embed(query_1)  →  pgvector.search(vec_1, top_k=20)  → 20 results
    ├── embed(query_2)  →  pgvector.search(vec_2, top_k=20)  → 20 results
    └── embed(query_3)  →  pgvector.search(vec_3, top_k=20)  → 20 results
    │
    └── All run via asyncio.gather() in parallel
    │
    └── Up to 80 results, each with {chunk_id, content, score, source}
```

**Why pgvector is not the only choice**: The `Retriever` protocol is all you need. Return `list[list[ScoredChunk]]` — one inner list per query — and the fusion stage handles the rest.

**Built-in**: `ConfigurablePgvectorRetriever`, `QdrantRetriever`, `ChromaDBRetriever`, `MemoryRetriever`.

**On failure**: One retriever times out → that retriever returns empty, others continue. All fail → fall back to sparse results (if enabled).

---

## Stage 2b: Sparse Retrieval (`Retriever` with `requires_embeddings=False`)

**Goal**: Keyword matching covers blind spots in vector search.

**Built-in BM25 implementation**:
- Pure Python Okapi BM25 (k1=1.5, b=0.75), zero dependencies
- Chinese: character bigram tokenization, "return" → "re" + "et" + "tu" + "ur" + "rn" + "return"
- English: whitespace tokenization + lowercase
- In-memory inverted index with async-safe double-buffered rebuild
- Incremental add/remove support

**Why BM25 is needed**:
- Vector search: "ASIN B09XYZ1234" — exact product code, embedding model may never have seen it
- BM25: "ASIN B09XYZ1234" — direct term match, guaranteed to find it

They complement each other: BM25 handles proper nouns, codes, typos; vector search handles synonyms, paraphrases, context.

**Replacing BM25**: Implement the `Retriever` protocol against Elasticsearch, Splade, or any other sparse engine.

**On failure**: BM25 not configured → not executed. Index load failure → skipped. Degradation recorded.

---

## Stage 3: Fusion (`Fusion`)

**Goal**: Merge results from multiple retrievers into a single ranked list.

**RRF formula** (default):
```
RRF_score(d) = Σ 1 / (k + rank_i(d))

where rank_i(d) is document d's position (1-indexed) in result list i
k=60 is the smoothing constant
```

**Hybrid scoring**:
```
Final_score = 0.8 × RRF(dense_results)
            + 0.2 × RRF(sparse_results)
```

With variant weight = 0.3, original query gets 0.7 weight and variants share 0.3.

**Why RRF instead of raw score normalization**:
- Vector search scores (cosine distance) and BM25 scores (term weight) are on different scales
- RRF only cares about rank position, not raw scores — naturally cross-engine comparable
- Smoothing constant k=60 reduces rank position influence differences

**Built-in**: `RRFFusion` (default), `WeightedFusion`, `RoundRobinFusion`.

**On failure**: Only one retriever has results → return those directly. Empty results → return empty list.

---

## Stage 4: Reranking (`Reranker`)

**Goal**: Fine-grained relevance judgment on fused candidates.

**Cross-Encoder vs Bi-Encoder**:

| Type | Encoding | Speed | Accuracy | Role in RAG |
|------|----------|-------|----------|-------------|
| Bi-Encoder | Encode Q and D separately, compute cosine | Fast (pre-computable) | Medium | Stage 2: Recall |
| Cross-Encoder | Concatenate Q+D, joint encode | Slow (per-pair) | High | Stage 4: Precision |

**Flow**:
```
Top-15 candidates (from fusion)
    │
    └── For each candidate, construct (query, document) pair
    │
    └── Cross-Encoder scores per pair: [0.92, 0.87, 0.65, 0.55, 0.33, ...]
    │
    └── Filter score < min_score: [0.92, 0.87, 0.65, 0.55]
    │
    └── Sort descending, take top_k: [0.92, 0.87, 0.65, 0.55]
```

**Built-in**: `CrossEncoderReranker` (local), `CohereReranker` (API).

**On failure**: Model load failure, inference OOM → skip reranking, keep fused Top-K. Degradation recorded.

---

## Stage 5: Context Building (`ContextBuilder`)

**Goal**: Return complete context, not isolated fragments.

**Problem**: Child chunk matching is precise but lacks context.

```
Returned to user: "Refund will be processed within 7 business days"
User confusion: "7 days starting from when?"

If parent chunk returned:
  "Return Policy
   1. Contact customer service within 30 days of receipt
   2. Customer service reviews and issues return label
   3. After returning, refund will be processed within 7 business days
   4. Refund goes back to original payment method"
```

**Implementation**:
1. From child chunk's `parent_chunk_id`, load parent content
2. Multiple child chunks from same parent → load parent only once
3. Greedy fill: add parents one by one until token count approaches `max_tokens`
4. Exceeds limit → truncate current parent with "..."

**Built-in**: `ParentExpander`, `WindowBuilder`, `PassthroughBuilder` (default — returns raw chunks).

**On failure**: Parent load fails (DB connection broken, parent deleted) → return child chunk text as-is.

---

## Inter-Stage State Flow

```
PipelineContext (mutable state bag passed through stages)

Stage 1 output: .expanded_queries + .entities
Bridge output:  .embeddings
Stage 2 output: .retriever_results (per-retriever, per-query ranked lists)
Stage 3 output: .fused_candidates (merged + parent-deduplicated)
Stage 4 output: .reranked_candidates (cross-encoder rescored)
Stage 5 output: .final_results (SearchResult objects with parent content)
```

Each stage reads from and writes to `PipelineContext`. The `Trace` object captures per-stage timing and degradation records for every search.

```
Trace
├── .query                  # original query
├── .total_ms               # wall-clock duration
├── .stage_timings[]        # per-stage {stage, elapsed_ms}
├── .degradations[]         # {stage, error} for failed stages
├── .degraded               # True if any stage failed
├── .metadata               # user-provided metadata
├── .expanded_queries[]     # Stage 1 variants (non-minimal level)
├── .entities{}             # Stage 1 extracted entities (non-minimal)
└── .retriever_hits{}       # per-retriever hit counts (non-minimal)
```

The three last fields are populated only when `trace_level != "minimal"`.
Override per request with `rag.search(q, trace_level="full")` for debugging.
