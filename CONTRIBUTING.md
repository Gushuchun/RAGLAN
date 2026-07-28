# Contributing to Raglan

## Development Setup

```bash
git clone https://github.com/Gushuchun/RAGLAN.git
cd raglan
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requirements: Python >= 3.10 (recommended: 3.12+).

## Quick Check

Before committing, run:

```bash
make check    # ruff + mypy + pytest
```

Or run steps individually:

```bash
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy raglan/
make test       # pytest -n auto
make cov        # pytest --cov --cov-report=html
```

## Running Tests

```bash
pytest                                    # all tests
pytest tests/unit/test_bm25.py           # single file
pytest -n auto                           # parallel (requires pytest-xdist)
pytest --cov=raglan --cov-report=html    # coverage report
pytest tests/ -m integration             # integration tests only
```

See [tests/README.md](tests/README.md) for the full test suite overview.

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check raglan/        # lint
ruff format raglan/       # format
ruff check --fix raglan/  # auto-fix
```

Type checking with mypy (strict mode):

```bash
mypy raglan/
```

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks (one-time)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Continuous Integration

Every pull request automatically runs:

| Check | Command |
|-------|---------|
| Lint | `ruff check .` |
| Type check | `mypy raglan/` |
| Tests | `pytest --cov --cov-fail-under=90` (Python 3.10–3.14) |
| Integration | Real PostgreSQL, ChromaDB, Qdrant via Docker services |
| Security | Bandit + pip-audit |
| Docs | README quickstart validation |

You can run the full CI pipeline locally with `make check`.

## Pull Request Process

1. **Discuss first** — for major changes, open an issue to discuss the design before coding.
2. Create a branch from `main`.
3. Write or update tests for your changes.
4. Ensure `make check` passes.
5. Update CHANGELOG.md under "Unreleased".
6. Open a PR using the template — describe what changed and why.

## Reporting Bugs

Open an issue with:
- Python version (`python --version`)
- Raglan version (`pip show raglan`)
- Minimal reproduction steps
- Full traceback if applicable

For feature requests, describe your use case and proposed API.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `refactor:` for code restructuring
- `test:` for test changes
- `chore:` for build/tooling changes

## Architecture

| Module | Purpose |
|--------|---------|
| `raglan/pipeline.py` | Pipeline engine + stage dispatch |
| `raglan/raglan.py` | Facade + Builder API |
| `raglan/protocols.py` | All user-implementable interfaces |
| `raglan/types.py` | Core data types (`ScoredChunk`, `Filter`, etc.) |
| `raglan/retrievers/` | Search backends (BM25, pgvector, Qdrant, ChromaDB, Memory) |
| `raglan/embedders/` | Text-to-vector embeddings (OpenAI, HuggingFace, DashScope) |
| `raglan/expanders/` | Query expansion (OpenAI, LiteLLM, Identity) |
| `raglan/fusion/` | Result fusion (RRF, Weighted, RoundRobin) |
| `raglan/rerankers/` | Cross-Encoder and Cohere reranking |
| `raglan/context_builders/` | Context assembly (Parent, Window, Passthrough) |
| `raglan/middleware/` | Timeout, Retry, CircuitBreaker, Logging, RateLimiter |
| `raglan/resilience/` | RetryBudget, HealthChecker |

See [docs/architecture.md](docs/architecture.md) and [docs/pipeline.md](docs/pipeline.md) for details.

## Questions?

Open a [Discussion](https://github.com/Gushuchun/RAGLAN/discussions).
