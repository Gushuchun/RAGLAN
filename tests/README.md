# Raglan Test Suite

## Structure

```
tests/
├── unit/              # Isolated module/class tests (mocks, no external deps)
├── integration/       # Multi-component + real backend tests (pgvector, ChromaDB, Qdrant)
├── e2e/               # End-to-end full pipeline tests with pre-computed embeddings
├── benchmark/         # Performance + memory stability tests
├── property/          # Hypothesis property-based tests
├── regression/        # Bug-fix regression tests
├── fixtures/          # Pre-computed test data (embeddings, queries, relevance)
├── conftest.py        # Shared pytest configuration
├── generate_e2e_fixtures.py  # One-time fixture generation script
└── README.md
```

## Running Tests

```bash
# All tests (parallel, excludes slow benchmarks)
pytest tests/ -n auto -k "not slow"

# Unit tests only
pytest tests/unit/ -n auto

# Integration tests (requires PostgreSQL, ChromaDB, Qdrant)
pytest tests/integration/ -m integration -n auto

# End-to-end tests
pytest tests/e2e/ -v -s -n auto

# Benchmark tests (single-process: xdist skews timing)
pytest tests/benchmark/ --benchmark-only

# Everything including slow tests
pytest tests/ -n auto
```

## Test Categories

| Directory | Marker | Requires |
|-----------|--------|----------|
| `unit/` | — | Nothing |
| `integration/` | `integration` | PostgreSQL, ChromaDB, Qdrant |
| `e2e/` | `integration` | Pre-computed fixtures |
| `benchmark/` | `slow` | PostgreSQL |
| `property/` | — | Nothing |

## Generating Fixtures

```bash
HF_ENDPOINT=https://hf-mirror.com python tests/generate_e2e_fixtures.py
```
