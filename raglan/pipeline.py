"""Pipeline engine — the central orchestrator that wires stages together.

A ``Pipeline`` is an ordered sequence of stages.  Stages grouped as a
``list`` inside the main sequence are executed concurrently (e.g.
multiple retrievers).  Middleware placed immediately before a stage
wraps it with cross-cutting behaviour (timeout, retry, etc.).
"""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Awaitable, Callable
from typing import Any

from raglan.exceptions import ConfigurationError
from raglan.types import (
    PipelineContext,
    SearchOptions,
    StageDegradation,
    StageTiming,
    Trace,
)


class Pipeline:
    """Ordered, observable sequence of retrieval pipeline stages.

    Parameters
    ----------
    stages:
        Ordered collection.  A ``list`` element means "run these
        concurrently".  Middleware placed before a stage wraps it.
    fallback_mode:
        ``"degrade"`` — on error, record a ``StageDegradation`` and
        continue.  ``"strict"`` — re-raise immediately.
    """

    def __init__(
        self,
        stages: list[Any],
        *,
        fallback_mode: str = "degrade",
        metrics_collector: Any | None = None,
        trace_level: str = "normal",
    ) -> None:
        # Pre-process: flatten middleware+stage pairs into _WrappedStage
        self._items = _preprocess(stages)
        self._fallback_mode = fallback_mode
        self._metrics = metrics_collector
        self._trace_level = trace_level

    def iter_stages(self) -> list[Any]:
        """Return a flat list of all stages (public API for resource cleanup).

        Middleware-wrapped stages are unwrapped — recursively, so a run of
        consecutive middleware (``[mw1, mw2, stage]``) still yields the
        underlying stage object.
        """
        result: list[Any] = []
        for item in self._items:
            stages = item if isinstance(item, list) else [item]
            for stage in stages:
                result.append(_unwrap_stage(stage))
        return result

    async def run(
        self,
        query: str,
        *,
        filters: list[Any] | None = None,
        options: SearchOptions | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        trace_level: str | None = None,
    ) -> tuple[list[Any], Trace]:
        """Execute all stages, returning ``(final_results, trace)``.

        Parameters
        ----------
        timeout:
            Global deadline in seconds for the entire pipeline run.
            When exceeded a ``TimeoutError`` degradation is recorded and
            whatever partial results are available are returned.
        trace_level:
            Per-request trace detail override: ``"minimal"``, ``"normal"``,
            or ``"full"``.  Defaults to the pipeline's configured level.
        """
        ctx = PipelineContext(
            query=query,
            filters=filters or [],
            options=options or SearchOptions(),
            metadata=metadata or {},
            started_at=_time.monotonic(),
        )

        # Per-request fallback override (SearchOptions.fallback_mode) wins;
        # otherwise fall back to the pipeline's configured mode.
        effective_fallback = ctx.options.fallback_mode or self._fallback_mode

        async def _run_all(ctx: PipelineContext) -> PipelineContext:
            for item in self._items:
                if isinstance(item, list):
                    ctx = await _run_parallel(item, ctx, effective_fallback)
                else:
                    ctx = await _run_single(item, ctx, effective_fallback)
            return ctx

        try:
            if timeout is not None:
                ctx = await asyncio.wait_for(_run_all(ctx), timeout=timeout)
            else:
                ctx = await _run_all(ctx)
        except (TimeoutError, asyncio.TimeoutError):
            ctx.degradations.append(
                StageDegradation(
                    stage="pipeline",
                    error=f"global timeout ({timeout:.1f}s) exceeded",
                )
            )

        # Apply top_k limit
        top_k = _opt(ctx.options.top_k, -1)
        if top_k > 0:
            ctx.final_results = ctx.final_results[:top_k]

        effective_trace_level = trace_level or self._trace_level
        trace = _build_trace(ctx, effective_trace_level)

        # Emit metrics
        if self._metrics is not None:
            await self._emit_metrics(ctx, trace)

        return ctx.final_results, trace

    async def _emit_metrics(self, ctx: PipelineContext, trace: Trace) -> None:
        """Send pipeline metrics to the configured collector."""
        assert self._metrics is not None
        await self._metrics.record_search(
            query=ctx.query,
            total_ms=trace.total_ms,
            result_count=len(ctx.final_results),
            degraded=trace.degraded,
            metadata=ctx.metadata,
        )
        for timing in trace.stage_timings:
            degradation = next((d for d in trace.degradations if d.stage == timing.stage), None)
            await self._metrics.record_stage(
                stage_name=timing.stage,
                elapsed_ms=timing.elapsed_ms,
                degraded=degradation is not None,
                error=degradation.error if degradation else None,
            )


# ============================================================================
# Stage execution helpers
# ============================================================================


async def _run_single(
    stage: Any,
    ctx: PipelineContext,
    fallback_mode: str,
) -> PipelineContext:
    """Execute one stage (or wrapped middleware+stage pair)."""
    name = _stage_name(stage)
    t0 = _time.monotonic()
    try:
        if isinstance(stage, _WrappedStage):
            ctx = await stage(ctx)
        else:
            ctx = await _dispatch_stage(stage, ctx)
    except Exception as exc:
        # In degrade mode any Exception subclass is caught so the pipeline
        # can continue.  MemoryError and RecursionError technically inherit
        # from Exception, but they are vanishingly rare in practice and
        # degrading them is an acceptable trade-off vs. a hard crash.
        ctx.degradations.append(StageDegradation(stage=name, error=str(exc)))
        if fallback_mode == "strict":
            raise
    finally:
        ctx.stage_timings.append(
            StageTiming(stage=name, elapsed_ms=(_time.monotonic() - t0) * 1000)
        )
    return ctx


async def _run_parallel(
    stages: list[Any],
    ctx: PipelineContext,
    fallback_mode: str,
) -> PipelineContext:
    """Execute a group of stages concurrently.

    All stages share the same *ctx* object and write results to distinct
    keys in ``ctx.retriever_results``.  The caller (``RaglanBuilder``) is
    responsible for ensuring each retriever has a unique ``.name`` so that
    results do not silently collide.
    """
    tasks = [_run_single(s, ctx, fallback_mode) for s in stages]
    await asyncio.gather(*tasks)
    return ctx


# ============================================================================
# Stage dispatch — routes ctx to the right protocol method
# ============================================================================

_StageHandler = Callable[[Any, PipelineContext], Awaitable[PipelineContext]]


async def _handle_expander(stage: Any, ctx: PipelineContext) -> PipelineContext:
    queries, entities = await stage.expand(ctx.query)
    # Protocol contract: the first element MUST be the original query so that
    # RRFFusion.original_query_idx weights it correctly. Auto-prepend if a
    # custom expander violates this, rather than silently corrupting fusion.
    if not queries or queries[0] != ctx.query:
        queries = [q for q in queries if q != ctx.query]
        queries.insert(0, ctx.query)
    ctx.expanded_queries = queries
    ctx.entities = entities
    return ctx


async def _handle_embedder(stage: Any, ctx: PipelineContext) -> PipelineContext:
    texts = ctx.expanded_queries if ctx.expanded_queries else [ctx.query]
    ctx.embeddings = await stage.embed(texts)
    return ctx


async def _handle_retriever(stage: Any, ctx: PipelineContext) -> PipelineContext:
    name = _stage_name(stage)
    # A sparse retriever does not require embeddings (e.g. BM25). Prefer the
    # protocol flag over name-matching so custom sparse retrievers work too.
    is_sparse = not bool(getattr(stage, "requires_embeddings", True))
    top_k = (
        _opt(ctx.options.bm25_top_k, _opt(ctx.options.dense_top_k, 20))
        if is_sparse
        else _opt(ctx.options.dense_top_k, 20)
    )
    results = await stage.retrieve(
        queries=ctx.expanded_queries if ctx.expanded_queries else [ctx.query],
        embeddings=ctx.embeddings,
        top_k=top_k,
        filters=ctx.filters if ctx.filters else None,
        timeout=ctx.options.retriever_timeout,
    )
    ctx.retriever_results[name] = results
    return ctx


async def _handle_fusion(stage: Any, ctx: PipelineContext) -> PipelineContext:
    ctx.fused_candidates = await stage.fuse(ctx.retriever_results)
    return ctx


async def _handle_reranker(stage: Any, ctx: PipelineContext) -> PipelineContext:
    candidates = ctx.fused_candidates or _flatten_retriever_results(ctx.retriever_results)
    ctx.reranked_candidates = await stage.rerank(
        query=ctx.query,
        candidates=candidates,
        top_k=_opt(ctx.options.reranker_top_n, _opt(ctx.options.top_k, 4)),
        min_score=max(_opt(ctx.options.reranker_min_score, 0.0), 0.0),
    )
    return ctx


async def _handle_context_builder(stage: Any, ctx: PipelineContext) -> PipelineContext:
    candidates = ctx.reranked_candidates or ctx.fused_candidates or []
    ctx.final_results = await stage.build(
        query=ctx.query,
        candidates=candidates,
        max_tokens=_opt(ctx.options.max_context_tokens, 6000),
    )
    return ctx


def _build_dispatch_table() -> dict[type, Any]:
    """Build a type→handler dispatch table for pipeline stages."""
    from raglan.protocols import (
        ContextBuilder,
        Embedder,
        Fusion,
        QueryExpander,
        Reranker,
        Retriever,
    )

    return {
        QueryExpander: _handle_expander,
        Embedder: _handle_embedder,
        Retriever: _handle_retriever,
        Fusion: _handle_fusion,
        Reranker: _handle_reranker,
        ContextBuilder: _handle_context_builder,
    }


async def _dispatch_stage(stage: Any, ctx: PipelineContext) -> PipelineContext:
    """Inspect the stage object and call the appropriate protocol method.

    Uses ``isinstance`` against runtime-checkable Protocols rather than
    ``hasattr`` duck-typing so that a custom object with an unrelated
    ``expand`` method is not accidentally routed as a QueryExpander.

    The dispatch table is built once at first call and cached.
    """

    # Look up or build the dispatch table (cached on the function)
    table: dict[type, _StageHandler] | None = getattr(_dispatch_stage, "_table", None)
    if table is None:
        table = _build_dispatch_table()
        _dispatch_stage._table = table  # type: ignore[attr-defined]

    for proto_type, handler in table.items():
        if isinstance(stage, proto_type):
            return await handler(stage, ctx)

    # Generic callable(ctx) -> ctx
    result: PipelineContext = await stage(ctx)
    return result


# ============================================================================
# Pre-processing: middleware + stage pairing
# ============================================================================


class _WrappedStage:
    """A middleware instance paired with the stage it wraps."""

    def __init__(self, middleware: Any, stage: Any) -> None:
        self._mw = middleware
        self._stage = stage
        self.name = _stage_name(stage)

    async def __call__(self, ctx: PipelineContext) -> PipelineContext:
        async def _next(c: PipelineContext) -> PipelineContext:
            _next.__name__ = self.name
            return await _dispatch_stage(self._stage, c)

        return await self._mw.wrap(ctx, _next)  # type: ignore[no-any-return]


def _unwrap_stage(stage: Any) -> Any:
    """Recursively unwrap ``_WrappedStage`` layers to the underlying stage."""
    while isinstance(stage, _WrappedStage):
        stage = stage._stage
    return stage


def _preprocess(items: list[Any]) -> list[Any]:
    """Flatten middleware+stage pairs into ``_WrappedStage`` objects.

    Supports a run of consecutive middleware wrapping a single stage —
    ``[mw1, mw2, stage]`` becomes ``_WrappedStage(mw1, _WrappedStage(mw2, stage))``
    so each middleware sees the next as its inner ``next`` callable.
    A middleware with nothing to wrap is a configuration error.
    """
    result: list[Any] = []
    i = 0
    while i < len(items):
        cur = items[i]
        if isinstance(cur, list):
            result.append(cur)
            i += 1
        elif _is_middleware(cur):
            # Collect the consecutive run of middleware.
            mws: list[Any] = [cur]
            j = i + 1
            while j < len(items) and _is_middleware(items[j]):
                mws.append(items[j])
                j += 1
            if j >= len(items):
                raise ConfigurationError(
                    f"Middleware {mws[-1].__class__.__name__} has no stage to wrap — "
                    "every middleware must be followed by a stage."
                )
            # Wrap from the innermost outward: the last middleware wraps the
            # stage, the previous middleware wraps that, and so on.
            wrapped: Any = items[j]
            for mw in reversed(mws):
                wrapped = _WrappedStage(mw, wrapped)
            result.append(wrapped)
            i = j + 1
        else:
            result.append(cur)
            i += 1
    return result


def _is_middleware(obj: Any) -> bool:
    from raglan.protocols import Middleware

    return isinstance(obj, Middleware)


# ============================================================================
# Helpers
# ============================================================================


def _stage_name(obj: Any) -> str:
    return str(getattr(obj, "name", obj.__class__.__name__))


def _opt(value: Any, default: Any) -> Any:
    """Return *value* if meaningful, else *default*.

    Sentinel values that trigger the default:
    - ``None``
    - Negative numbers (``-1`` is the conventional "use default" sentinel)
    - Empty strings

    Zero is treated as a valid, explicit value (e.g. ``top_k=0`` means
    "return zero results").
    """
    if value is None:
        return default
    if isinstance(value, (int, float)) and value < 0:
        return default
    if isinstance(value, str) and not value:
        return default
    return value


def _flatten_retriever_results(
    results: dict[str, list[list[Any]]],
) -> list[Any]:
    """Flatten per-retriever per-query results into a single candidate list."""
    flat: list[Any] = []
    for query_results in results.values():
        for chunk_list in query_results:
            flat.extend(chunk_list)
    return flat


def _build_trace(ctx: PipelineContext, trace_level: str = "normal") -> Trace:
    total = (_time.monotonic() - ctx.started_at) * 1000

    # Apply trace level filtering for security:
    # - minimal: timings + counts only, no query text, no metadata
    # - normal:  + degradation records, per-stage metadata (default)
    # - full:    + raw intermediate results (debug only)
    query = ctx.query if trace_level != "minimal" else ""
    metadata = ctx.metadata if trace_level != "minimal" else {}
    degradations = ctx.degradations if trace_level != "minimal" else []
    expanded_queries = ctx.expanded_queries if trace_level != "minimal" else []
    entities = ctx.entities if trace_level != "minimal" else {}

    # Per-retriever hit counts (useful for diagnosing coverage).
    retriever_hits = (
        {
            name: sum(len(per_query) for per_query in query_results)
            for name, query_results in ctx.retriever_results.items()
        }
        if trace_level != "minimal"
        else {}
    )

    return Trace(
        query=query,
        total_ms=total,
        stage_timings=ctx.stage_timings,
        degradations=degradations,
        metadata=metadata,
        expanded_queries=expanded_queries,
        entities=entities,
        retriever_hits=retriever_hits,
    )
