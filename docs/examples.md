# Multi-Scenario Examples

## Example 0: Quick Start (Direct Instantiation)

The simplest entry points — a retriever as the first positional argument,
everything else optional:

```python
import asyncio
from raglan import Raglan
from raglan.retrievers import BM25Retriever


async def main():
    bm25 = BM25Retriever()

    async def chunks():
        yield [("doc1", "Return policy: 30 days.", None)]

    await bm25.index(chunks())

    # Direct instantiation — BM25-only, zero extra config.
    rag = Raglan([bm25])
    results, trace = await rag.search("how to return")

    # Incremental configuration — add stages step by step.
    rag2 = Raglan()
    rag2.add_retriever(bm25)
    rag2.set_embedder("openai:text-embedding-3-small")  # string shorthand
    rag2.set_expander("openai:gpt-4o-mini")
    rag2.set_fusion("rrf")

    # Config template — fill in a defaults dict.
    cfg = Raglan.config()
    cfg["retrievers"].append({"type": "bm25"})
    rag3 = Raglan.from_config(cfg)


asyncio.run(main())
```

## Example 1: pgvector + OpenAI Embeddings

```python
import asyncio
from raglan import Raglan
from raglan.embedders import OpenAIEmbedder
from raglan.retrievers import ConfigurablePgvectorRetriever, BM25Retriever


async def load_parents(chunk_ids: list[str]) -> dict[str, str]:
    # Your parent-chunk loading logic
    ...


rag = (
    Raglan.builder()
    .with_embedder(OpenAIEmbedder(model="text-embedding-3-small"))
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://user:pass@localhost/db",
                table="document_chunks",
                parent_id_column="parent_id",
            ),
            BM25Retriever(),
        ]
    )
    .with_context_builder(ParentExpander(loader=load_parents))
    .build()
)

results, trace = await rag.search("damaged order return policy")
for r in results:
    print(f"[{r.score:.3f}] {r.content[:100]}...")
```

## Example 2: Fully Offline (No LLM API calls)

For privacy-sensitive or air-gapped deployments:

```python
from raglan.embedders import HuggingFaceEmbedder
from raglan.retrievers import BM25Retriever
from raglan.retrievers import MemoryRetriever

rag = (
    Raglan.builder()
    .with_embedder(
        HuggingFaceEmbedder(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
        )
    )
    .with_retrievers([MemoryRetriever(), BM25Retriever()])
    # No expander → uses IdentityExpander (no LLM calls)
    # No reranker → skip cross-encoder
    .build()
)
```

## Example 3: Chinese Environment + BGE Reranker

```python
from raglan.expanders import OpenAIExpander
from raglan.rerankers import CrossEncoderReranker

rag = (
    Raglan.builder()
    .with_expander(
        OpenAIExpander(
            model="gpt-4o-mini",
            prompt_template="""Generate {num_queries} Chinese search query variants covering different aspects.
Original query: {query}
Return JSON only: {{"variants": ["variant1", "variant2", ...]}}""",
        )
    )
    .with_embedder(OpenAIEmbedder(model="text-embedding-3-small"))
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://...",
                table="kb.chunks",
                embedding_column="embedding",
            ),
            BM25Retriever(),
        ]
    )
    .with_reranker(
        CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu",
            min_score=0.5,
        )
    )
    .build()
)
```

## Example 4: LiteLLM Multi-Provider Expansion

Use any LLM provider for query expansion: OpenAI, Anthropic, Azure, Ollama, etc.

```python
from raglan.expanders import LiteLLMExpander

rag = (
    Raglan.builder()
    .with_expander(
        LiteLLMExpander(
            model="ollama/llama3",  # Local Ollama model
            api_base="http://localhost:11434",
        )
    )
    .with_retrievers([BM25Retriever()])
    .build()
)

# Or with Anthropic Claude:
rag = (
    Raglan.builder()
    .with_expander(
        LiteLLMExpander(
            model="claude-3-haiku-20240307",
            api_key="sk-ant-xxx",
        )
    )
    .with_retrievers([...])
    .build()
)
```

## Example 5: Cohere Rerank API

```python
from raglan.rerankers import CohereReranker

rag = (
    Raglan.builder()
    .with_embedder(OpenAIEmbedder())
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://...",
                table="kb.chunks",
                embedding_column="embedding",
            ),
        ]
    )
    .with_reranker(
        CohereReranker(
            model="rerank-v3.5",
            min_score=0.3,
            api_key="your-cohere-key",
        )
    )
    .build()
)
```

## Example 6: Alibaba DashScope Embeddings

```python
from raglan.embedders import DashScopeEmbedder

rag = (
    Raglan.builder()
    .with_embedder(
        DashScopeEmbedder(
            model="text-embedding-v3",
            api_key="sk-xxx",
        )
    )
    .with_retrievers(
        [
            ConfigurablePgvectorRetriever(
                connection_string="postgresql://...",
                table="kb.chunks",
                embedding_column="embedding",
            ),
        ]
    )
    .build()
)
```

## Example 7: Custom Retriever for Proprietary Knowledge Base

```python
class MyKnowledgeBaseRetriever:
    name = "my_kb"
    requires_embeddings = True

    async def retrieve(self, queries, embeddings, top_k, filters=None, timeout=None):
        results = []
        for emb in embeddings:
            rows = await my_proprietary_db.vector_search(emb, top_k, filters)
            results.append(
                [
                    ScoredChunk(
                        chunk_id=row["doc_uuid"],
                        content=row["body"],
                        score=row["similarity"],
                        parent_chunk_id=row.get("section_id"),
                        chunk_metadata={"lang": row["lang"], "team": row["owner"]},
                        source=self.name,
                    )
                    for row in rows
                ]
            )
        return results

    async def index(self, chunks): ...
    async def add(self, chunks): ...
    async def remove(self, chunk_ids): ...


rag = Raglan.builder().with_retrievers([MyKnowledgeBaseRetriever()]).build()
```

## Example 8: Pipeline with Middleware

```python
from raglan.pipeline import Pipeline
from raglan.middleware import TimeoutMiddleware, RetryMiddleware

pipeline = Pipeline(
    [
        TimeoutMiddleware(30.0),
        OpenAIExpander(),  # 30s timeout for LLM call
        RetryMiddleware(max_retries=3),
        OpenAIEmbedder(),  # Retry embedding on failure
        [BM25Retriever(), PgvectorRetriever(...)],  # Parallel retrieval
        RRFFusion(),
        CrossEncoderReranker(),
        PassthroughBuilder(),
    ]
)

rag = Raglan(pipeline)
```

## Example 9: Debugging with Trace

```python
results, trace = await rag.search("how to return")

print(f"Total: {trace.total_ms:.1f}ms")
for timing in trace.stage_timings:
    print(f"  {timing.stage}: {timing.elapsed_ms:.1f}ms")

if trace.degraded:
    for d in trace.degradations:
        print(f"  DEGRADED: [{d.stage}] {d.error}")
```

## Example 10: Serialization Round-Trip

```python
# Save
rag = Raglan.builder().with_retrievers([BM25Retriever(k1=1.2, b=0.5)]).build()
config = rag.export_config()
# {"fallback_mode": "degrade",
#  "retrievers": [{"type": "bm25", "params": {"k1": 1.2, "b": 0.5}}]}

import json

with open("pipeline.json", "w") as f:
    json.dump(config, f)

# Load
with open("pipeline.json") as f:
    config = json.load(f)
rag = Raglan.from_dict(config)
```

## Example 11: Metrics Collection

```python
from raglan.observability import LoggingMetricsCollector

rag = (
    Raglan.builder()
    .with_retrievers([BM25Retriever()])
    .with_metrics_collector(LoggingMetricsCollector())
    .build()
)

# Each search now logs stage timing and degradation to the Python logger
results, trace = await rag.search("test query")
```

## Example 12: Health Checks

```python
from raglan.resilience import HealthChecker

health = HealthChecker(
    {
        "pgvector": lambda: check_postgres_connection(),
        "openai": lambda: check_openai_api(),
    }
)
statuses = await health.check_all(timeout=5.0)
for name, status in statuses.items():
    print(f"{name}: {'OK' if status.healthy else 'FAILED'} ({status.latency_ms:.1f}ms)")
```
