.PHONY: help install test lint typecheck format clean build cov

PYTHON := python
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PYTEST := $(PYTHON) -m pytest

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install in development mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests (parallel)
	$(PYTEST) tests/ -v -n auto

cov:  ## Run tests with coverage (fail under 90%, parallel)
	$(PYTEST) tests/ --cov=raglan --cov-report=term-missing --cov-report=html --cov-fail-under=90 -n auto

lint:  ## Run ruff linter
	$(RUFF) check .

format:  ## Auto-fix ruff issues
	$(RUFF) check . --fix

typecheck:  ## Run mypy type checker
	$(MYPY) raglan/ --ignore-missing-imports

check: lint typecheck test  ## Run all checks (lint + typecheck + test)

build:  ## Build wheel and sdist
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ __pycache__/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
