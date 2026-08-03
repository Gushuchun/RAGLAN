"""Raglan facade and Builder — the primary user-facing API.

This module provides two ways to construct a Raglan pipeline:

1. **Facade** — two lines for prototyping::

       rag = Raglan(pg_conn, api_key)
       results, trace = await rag.search("how to return my order")

2. **Builder** — explicit, typed configuration for production::

       rag = (
           Raglan.builder()
           .with_expander(OpenAIExpander(...))
           .with_retrievers([PgvectorRetriever(...), BM25Retriever()])
           .with_reranker(CrossEncoderReranker(...))
           .build()
       )
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from raglan.context_builders.passthrough import PassthroughBuilder
from raglan.exceptions import ConfigurationError
from raglan.expanders.identity import IdentityExpander
from raglan.fusion.rrf import RRFFusion
from raglan.pipeline import Pipeline
from raglan.protocols import (
    ContextBuilder,
    Embedder,
    Fusion,
    QueryExpander,
    Reranker,
    Retriever,
)
from raglan.types import (
    Filter,
    SearchOptions,
    SearchResult,
    Trace,
)


class Raglan:
    """The main Raglan entry point — a configured retrieval pipeline.

    Three ways to construct:

    1. **Direct instantiation** (recommended) — pass a retriever or list as
       the first positional argument, then configure incrementally::

           rag = Raglan([bm25])
           rag.set_embedder("openai:text-embedding-3-small")
           rag.set_expander("openai:gpt-4o-mini")
           results, trace = await rag.search("my query")

    2. **Empty instance + incremental setters** — build up piece by piece::

           rag = Raglan()
           rag.add_retriever(bm25)
           rag.set_fusion("rrf")

    3. **Builder** — explicit, typed configuration for production::

           rag = Raglan.builder().with_retrievers([bm25]).build()

    Supports ``async with`` for automatic resource cleanup::

        async with Raglan.builder().with_retrievers([...]).build() as rag:
            results, trace = await rag.search("my query")
    """

    def __init__(
        self,
        retrievers: Retriever | list[Retriever] | Pipeline | None = None,
        *,
        expander: Any = None,
        embedder: Any = None,
        fusion: Any = None,
        reranker: Any = None,
        context_builder: Any = None,
        fallback_mode: str = "degrade",
        trace_level: str = "normal",
        metrics_collector: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._pipeline: Pipeline | None
        self._config: dict[str, Any]
        self._builder: RaglanBuilder | None

        if isinstance(retrievers, Pipeline):
            # Legacy path: Raglan(pipeline, config=...) — a pre-built pipeline.
            self._pipeline = retrievers
            self._config = config or {}
            self._builder = None
            return

        # Configurable path: hold an internal Builder, assemble lazily.
        self._builder = RaglanBuilder()
        self._pipeline = None
        self._config = config or {}

        if retrievers is not None:
            if isinstance(retrievers, list):
                for r in retrievers:
                    self._builder.add_retriever(r)
            else:
                self._builder.add_retriever(retrievers)

        if expander is not None:
            self.set_expander(expander)
        if embedder is not None:
            self.set_embedder(embedder)
        if fusion is not None:
            self.set_fusion(fusion)
        if reranker is not None:
            self.set_reranker(reranker)
        if context_builder is not None:
            self.set_context_builder(context_builder)
        self.set_fallback_mode(fallback_mode)
        self.set_trace_level(trace_level)
        if metrics_collector is not None:
            self._builder.with_metrics_collector(metrics_collector)

    # ------------------------------------------------------------------
    # Incremental configuration
    # ------------------------------------------------------------------

    def _ensure_mutable(self) -> None:
        """Raise if the pipeline has already been built (immutable after search)."""
        if self._pipeline is not None:
            raise ConfigurationError(
                "Raglan pipeline already built — configure it before the first search."
            )

    def add_retriever(self, retriever: Retriever) -> Raglan:
        """Add a retriever. Returns ``self`` for chaining."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.add_retriever(retriever)
        return self

    def add_retrievers(self, retrievers: list[Retriever]) -> Raglan:
        """Add multiple retrievers. Returns ``self`` for chaining."""
        for r in retrievers:
            self.add_retriever(r)
        return self

    def set_expander(self, expander: Any) -> Raglan:
        """Set the query expander (object, ``"vendor:model"`` string, or dict)."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_expander(_coerce_component(expander, "expander"))
        return self

    def set_embedder(self, embedder: Any) -> Raglan:
        """Set the embedder (object, ``"vendor:model"`` string, or dict)."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_embedder(_coerce_component(embedder, "embedder"))
        return self

    def set_fusion(self, fusion: Any) -> Raglan:
        """Set the fusion strategy (object, ``"rrf"``/``"weighted"``, or dict)."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_fusion(_coerce_component(fusion, "fusion"))
        return self

    def set_reranker(self, reranker: Any) -> Raglan:
        """Set the reranker (object, string, or dict). ``None`` disables reranking."""
        self._ensure_mutable()
        assert self._builder is not None
        if reranker is not None:
            self._builder.with_reranker(_coerce_component(reranker, "reranker"))
        return self

    def set_context_builder(self, cb: Any) -> Raglan:
        """Set the context builder (object, string, or dict)."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_context_builder(_coerce_component(cb, "context_builder"))
        return self

    def set_fallback_mode(self, mode: str) -> Raglan:
        """Set fallback behaviour: ``"degrade"`` or ``"strict"``."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_fallback_mode(mode)
        return self

    def set_trace_level(self, level: str) -> Raglan:
        """Set trace detail: ``"minimal"``, ``"normal"``, or ``"full"``."""
        self._ensure_mutable()
        assert self._builder is not None
        self._builder.with_trace_level(level)
        return self

    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Pipeline:
        """Lazily build and return the underlying pipeline."""
        if self._pipeline is None:
            assert self._builder is not None
            rag = self._builder.build()
            self._pipeline = rag._pipeline
            self._config = rag._config
        return self._pipeline  # type: ignore[return-value]

    async def __aenter__(self) -> Raglan:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all resources held by retrievers in the pipeline.

        Iterates over stages via the public ``Pipeline.iter_stages()``
        API and calls ``close()`` on any stage that has one (pgvector
        pools, Qdrant clients, etc.).  Each stage is closed independently
        — a failure in one does not prevent other stages from being closed.
        """
        if self._pipeline is None:
            return
        for stage in self._pipeline.iter_stages():
            closer = getattr(stage, "close", None)
            if closer is not None:
                with contextlib.suppress(Exception):
                    await closer()

    async def warm_up(self) -> None:
        """Pre-load any stage that supports it (rerankers, embedders).

        Call once during application startup to avoid a slow first request
        (e.g. model download for a Cross-Encoder reranker).
        """
        for stage in self._get_pipeline().iter_stages():
            warmer = getattr(stage, "warm_up", None)
            if warmer is not None:
                await warmer()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: list[Filter] | None = None,
        options: SearchOptions | None = None,
        metadata: dict[str, Any] | None = None,
        trace_level: str | None = None,
    ) -> tuple[list[SearchResult], Trace]:
        """Run a single search and return ``(results, trace)``.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Override the pipeline-wide ``top_k`` for this request.
        filters:
            Metadata filters applied at the retriever level.
        options:
            Per-request option overrides (see ``SearchOptions``).
        metadata:
            Arbitrary metadata attached to the trace.
        trace_level:
            Per-request trace detail override: ``"minimal"``, ``"normal"``,
            or ``"full"``.  Defaults to the pipeline's configured level.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if len(query) > 10000:
            raise ValueError(
                f"query exceeds maximum length of 10000 characters "
                f"(got {len(query)}). Consider truncating your input."
            )
        opts = options or SearchOptions()
        if top_k is not None:
            opts.top_k = top_k
        return await self._get_pipeline().run(
            query,
            filters=filters,
            options=opts,
            metadata=metadata,
            trace_level=trace_level,
        )

    async def batch_search(
        self,
        queries: list[str],
        *,
        top_k: int | None = None,
        filters: list[Filter] | None = None,
        options: SearchOptions | None = None,
        metadata: dict[str, Any] | None = None,
        max_concurrency: int = 10,
    ) -> list[tuple[list[SearchResult], Trace]]:
        """Run multiple searches with bounded concurrency.

        Parameters
        ----------
        queries:
            List of search queries to execute.
        top_k:
            Override the pipeline-wide ``top_k`` for all queries.
        filters:
            Metadata filters applied to each query.
        options:
            Per-request option overrides (see ``SearchOptions``).
        metadata:
            Arbitrary metadata attached to each trace.
        max_concurrency:
            Maximum number of concurrent searches.  Default 10.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(q: str) -> tuple[list[SearchResult], Trace]:
            async with semaphore:
                return await self.search(
                    q, top_k=top_k, filters=filters, options=options, metadata=metadata
                )

        return await asyncio.gather(*(_bounded(q) for q in queries))

    def search_sync(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: list[Filter] | None = None,
        options: SearchOptions | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[SearchResult], Trace]:
        """Synchronous wrapper for environments without an event loop."""
        return asyncio.run(
            self.search(query, top_k=top_k, filters=filters, options=options, metadata=metadata)
        )

    # ------------------------------------------------------------------
    # Builder entry points
    # ------------------------------------------------------------------

    @staticmethod
    def builder() -> RaglanBuilder:
        """Create a ``RaglanBuilder`` to assemble a pipeline step by step."""
        return RaglanBuilder()

    @staticmethod
    def from_dict(config: dict[str, Any]) -> Raglan:
        """Construct a Raglan instance from a configuration dictionary.

        The dictionary format mirrors the Builder API.  Example::

            {
                "expander": {"type": "openai", "model": "gpt-4o-mini"},
                "retrievers": [{"type": "bm25"}, {"type": "pgvector", "conn": "..."}],
                "fusion": {"type": "rrf"},
                "fallback_mode": "degrade",
            }

        Keys not present in the dictionary use their defaults.
        """
        return RaglanBuilder._from_dict(config).build()

    @staticmethod
    def config() -> dict[str, Any]:
        """Return a configuration template with sensible defaults.

        Fill in the ``retrievers`` list — the **only required key** — and any
        other keys you wish to override, then pass the result to
        :meth:`from_config`.  Calling ``from_config(Raglan.config())`` with
        an empty ``retrievers`` list raises ``ConfigurationError``.

        The template is JSON/YAML-serialisable::

            cfg = Raglan.config()
            cfg["retrievers"].append({"type": "pgvector", "params": {...}})
            cfg["expander"] = "openai:gpt-4o-mini"
            rag = Raglan.from_config(cfg)
        """
        return {
            "retrievers": [],
            "expander": None,
            "embedder": None,
            "fusion": "rrf",
            "reranker": None,
            "context_builder": None,
            "fallback_mode": "degrade",
            "trace_level": "normal",
        }

    @staticmethod
    def from_config(config: dict[str, Any]) -> Raglan:
        """Construct a Raglan instance from a config-template dictionary.

        Accepts the same shape as :meth:`from_dict`; ``None`` values are
        treated as "use the default".  This is an alias for :meth:`from_dict`
        — the two are interchangeable.
        """
        return Raglan.from_dict(config)

    def export_config(self) -> dict[str, Any]:
        """Export the current configuration as a serialisable dictionary.

        The returned dict is a deep copy — safe to mutate without affecting
        the running pipeline.  It can be saved (e.g. as JSON) and later
        passed to ``Raglan.from_dict()`` to recreate an equivalent pipeline.
        Note that custom callables and externally-managed objects (e.g.
        database connections) are represented as markers.
        """
        if self._pipeline is None and self._builder is not None:
            # Incremental instance not yet built — serialise the builder state.
            return self._builder._to_config_dict()
        return dict(self._config)

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`export_config`."""
        return self.export_config()


# ============================================================================
# Component registry — maps type-name strings to classes for from_dict()
# ============================================================================


def _build_component_registry() -> dict[str, type]:
    """Build a registry of known component types for deserialisation."""
    from raglan.context_builders.parent_expander import ParentExpander
    from raglan.context_builders.passthrough import PassthroughBuilder
    from raglan.context_builders.window import WindowBuilder
    from raglan.embedders.huggingface import HuggingFaceEmbedder
    from raglan.embedders.openai import OpenAIEmbedder
    from raglan.expanders.identity import IdentityExpander
    from raglan.expanders.openai import OpenAIExpander
    from raglan.fusion.round_robin import RoundRobinFusion
    from raglan.fusion.rrf import RRFFusion
    from raglan.fusion.weighted import WeightedFusion
    from raglan.rerankers.cross_encoder import CrossEncoderReranker
    from raglan.retrievers.bm25 import BM25Retriever
    from raglan.retrievers.memory import MemoryRetriever

    registry: dict[str, type] = {
        "identity": IdentityExpander,
        "openai_expander": OpenAIExpander,
        "openai_embedder": OpenAIEmbedder,
        "huggingface_embedder": HuggingFaceEmbedder,
        "bm25": BM25Retriever,
        "memory": MemoryRetriever,
        "rrf": RRFFusion,
        "weighted": WeightedFusion,
        "round_robin": RoundRobinFusion,
        "cross_encoder": CrossEncoderReranker,
        "passthrough": PassthroughBuilder,
        "parent_expander": ParentExpander,
        "window": WindowBuilder,
    }

    # Optional dependencies — only register if importable
    try:
        from raglan.retrievers.configurable_pgvector import (
            ConfigurablePgvectorRetriever,
        )

        registry["pgvector"] = ConfigurablePgvectorRetriever
    except ImportError:
        pass

    try:
        from raglan.retrievers.chromadb import ChromaDBRetriever

        registry["chromadb"] = ChromaDBRetriever
    except ImportError:
        pass

    try:
        from raglan.retrievers.qdrant import QdrantRetriever

        registry["qdrant"] = QdrantRetriever
    except ImportError:
        pass

    try:
        from raglan.expanders.litellm import LiteLLMExpander

        registry["litellm_expander"] = LiteLLMExpander
    except ImportError:
        pass

    try:
        from raglan.embedders.dashscope import DashScopeEmbedder

        registry["dashscope_embedder"] = DashScopeEmbedder
    except ImportError:
        pass

    try:
        from raglan.rerankers.cohere import CohereReranker

        registry["cohere_reranker"] = CohereReranker
    except ImportError:
        pass

    return registry


_COMPONENT_REGISTRY = _build_component_registry()


def register_component(type_name: str, cls: type) -> None:
    """Register a custom component class for ``from_dict()``/``from_config()``.

    After registration, configuration dictionaries can reference the
    component by type name::

        from raglan import register_component

        register_component("my_retriever", MyRetriever)
        rag = Raglan.from_dict(
            {"retrievers": [{"type": "my_retriever", "params": {...}}]}
        )

    The class must be constructible from its ``params`` kwargs (and should
    implement ``to_dict()`` to round-trip through ``export_config()``).
    """
    if not isinstance(type_name, str) or not type_name:
        raise ConfigurationError("type_name must be a non-empty string")
    if type_name in _COMPONENT_REGISTRY:
        raise ConfigurationError(f"Component type '{type_name}' is already registered.")
    _COMPONENT_REGISTRY[type_name] = cls


# ============================================================================
# Builder
# ============================================================================


class RaglanBuilder:
    """Step-by-step assembler for a Raglan pipeline.

    Call ``.build()`` to validate the configuration and produce a
    ``Raglan`` instance.  Every ``with_*`` method returns ``self`` for
    method chaining.
    """

    def __init__(self) -> None:
        self._expander: QueryExpander | None = None
        self._embedder: Embedder | None = None
        self._retrievers: list[Retriever] = []
        self._fusion: Fusion | None = None
        self._reranker: Reranker | None = None
        self._context_builder: ContextBuilder | None = None
        self._metrics_collector: Any | None = None
        self._trace_level: str = "normal"
        self._fallback_mode: str = "degrade"

    # -- stage setters --------------------------------------------------------

    def with_expander(self, expander: QueryExpander) -> RaglanBuilder:
        """Attach a query expander (Stage 1)."""
        self._expander = expander
        return self

    def with_embedder(self, embedder: Embedder) -> RaglanBuilder:
        """Attach an embedder (Stage 1→2 bridge). Required for dense retrievers."""
        self._embedder = embedder
        return self

    def with_retrievers(self, retrievers: list[Retriever]) -> RaglanBuilder:
        """Attach one or more retrievers (Stage 2). At least one is required.

        Replaces any previously configured retrievers.
        """
        self._retrievers = retrievers
        return self

    def add_retriever(self, retriever: Retriever) -> RaglanBuilder:
        """Append a single retriever to the current list. Returns ``self``."""
        self._retrievers.append(retriever)
        return self

    def with_fusion(self, fusion: Fusion) -> RaglanBuilder:
        """Attach a fusion strategy (Stage 3). Defaults to ``RRFFusion``."""
        self._fusion = fusion
        return self

    def with_reranker(self, reranker: Reranker) -> RaglanBuilder:
        """Attach a reranker (Stage 4). Optional — omit to skip reranking."""
        self._reranker = reranker
        return self

    def with_context_builder(self, cb: ContextBuilder) -> RaglanBuilder:
        """Attach a context builder (Stage 5). Defaults to ``PassthroughBuilder``."""
        self._context_builder = cb
        return self

    def with_fallback_mode(self, mode: str) -> RaglanBuilder:
        if mode not in ("degrade", "strict"):
            raise ConfigurationError(f"fallback_mode must be 'degrade' or 'strict', got '{mode}'")
        self._fallback_mode = mode
        return self

    def with_trace_level(self, level: str) -> RaglanBuilder:
        """Set the trace detail level.

        ``"minimal"`` — timings + counts only (safe for production logs).
        ``"normal"`` — + degradation records + metadata (default).
        ``"full"`` — + raw intermediate results (debug only).
        """
        if level not in ("minimal", "normal", "full"):
            raise ConfigurationError(
                f"trace_level must be 'minimal', 'normal', or 'full', got '{level}'"
            )
        self._trace_level = level
        return self

    def with_metrics_collector(self, collector: Any) -> RaglanBuilder:
        """Attach a ``MetricsCollector`` for observability."""
        self._metrics_collector = collector
        return self

    # -- build ----------------------------------------------------------------

    def build(self) -> Raglan:
        """Validate and assemble the pipeline."""
        self._validate()

        # Ensure unique retriever names so parallel stages don't collide
        self._deduplicate_retriever_names()

        # Apply defaults
        expander = self._expander or IdentityExpander()
        fusion: Fusion = self._fusion or RRFFusion()
        context_builder: ContextBuilder = self._context_builder or PassthroughBuilder()

        # Assemble stages
        stages: list[Any] = [expander]

        if self._embedder is not None:
            stages.append(self._embedder)

        if len(self._retrievers) == 1:
            stages.append(self._retrievers[0])
        else:
            stages.append(self._retrievers)  # parallel group

        stages.append(fusion)

        if self._reranker is not None:
            stages.append(self._reranker)

        stages.append(context_builder)

        config = self._to_config_dict()
        pipeline = Pipeline(
            stages,
            fallback_mode=self._fallback_mode,
            metrics_collector=self._metrics_collector,
            trace_level=self._trace_level,
        )
        return Raglan(pipeline, config=config)

    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if not self._retrievers:
            raise ConfigurationError("At least one Retriever is required.")

        dense_retrievers = [r for r in self._retrievers if r.requires_embeddings]
        if dense_retrievers and self._embedder is None:
            raise ConfigurationError(
                f"{len(dense_retrievers)} retriever(s) require embeddings "
                f"({[r.name for r in dense_retrievers]}), but no Embedder "
                f"was configured."
            )

    def _deduplicate_retriever_names(self) -> None:
        """Ensure every retriever has a unique ``.name``.

        When multiple retrievers share the same name, parallel execution
        in the pipeline would silently overwrite each other's results.
        This method appends a numeric suffix (``_2``, ``_3``, …) to
        duplicates so each retriever writes to a distinct key in
        ``ctx.retriever_results``.
        """
        seen: dict[str, int] = {}
        for r in self._retrievers:
            base = r.name
            if base in seen:
                seen[base] += 1
                r.name = f"{base}_{seen[base]}"
            else:
                seen[base] = 1

    @classmethod
    def _from_dict(cls, config: dict[str, Any]) -> RaglanBuilder:
        """Construct a Builder from a configuration dictionary."""
        builder = cls()

        if "fallback_mode" in config:
            builder.with_fallback_mode(config["fallback_mode"])

        if "trace_level" in config:
            builder.with_trace_level(config["trace_level"])

        if "metrics" in config:
            builder.with_metrics_collector(_instantiate(config["metrics"], "metrics"))

        # Expander
        expander_cfg = config.get("expander")
        if expander_cfg:
            builder.with_expander(_coerce_component(expander_cfg, "expander"))

        # Embedder
        embedder_cfg = config.get("embedder")
        if embedder_cfg:
            builder.with_embedder(_coerce_component(embedder_cfg, "embedder"))

        # Retrievers
        retriever_cfgs = config.get("retrievers", [])
        if retriever_cfgs:
            retrievers = [_coerce_component(c, "retriever") for c in retriever_cfgs]
            builder.with_retrievers(retrievers)

        # Fusion
        fusion_cfg = config.get("fusion")
        if fusion_cfg:
            builder.with_fusion(_coerce_component(fusion_cfg, "fusion"))

        # Reranker
        reranker_cfg = config.get("reranker")
        if reranker_cfg:
            builder.with_reranker(_coerce_component(reranker_cfg, "reranker"))

        # Context builder
        ctx_cfg = config.get("context_builder")
        if ctx_cfg:
            builder.with_context_builder(_coerce_component(ctx_cfg, "context_builder"))

        return builder

    def _to_config_dict(self) -> dict[str, Any]:
        """Serialize the current builder configuration to a dictionary."""
        config: dict[str, Any] = {
            "fallback_mode": self._fallback_mode,
            "trace_level": self._trace_level,
        }

        # Expander (only serialize non-default)
        if self._expander is not None and not _is_identity_expander(self._expander):
            config["expander"] = _serialize(self._expander)

        # Embedder
        if self._embedder is not None:
            config["embedder"] = _serialize(self._embedder)

        # Retrievers
        if self._retrievers:
            config["retrievers"] = [_serialize(r) for r in self._retrievers]

        # Fusion (only if non-default)
        if self._fusion is not None:
            config["fusion"] = _serialize(self._fusion)

        # Reranker
        if self._reranker is not None:
            config["reranker"] = _serialize(self._reranker)

        # Context builder (only if non-default)
        if self._context_builder is not None and not _is_passthrough_builder(self._context_builder):
            config["context_builder"] = _serialize(self._context_builder)

        # Metrics collector (only if non-default)
        if self._metrics_collector is not None and hasattr(self._metrics_collector, "name"):
            config["metrics"] = _serialize(self._metrics_collector)

        return config


# ============================================================================
# Serialization helpers
# ============================================================================


def _serialize(component: Any) -> dict[str, Any]:
    """Serialize a pipeline component to a ``{type, params}`` dict.

    Components must implement ``to_dict()``.  Raising instead of silently
    emitting ``{"type": name, "params": {}}`` prevents a config that would
    fail later in ``from_dict()`` from being produced undetected.
    """
    if hasattr(component, "to_dict"):
        return component.to_dict()  # type: ignore[no-any-return]
    type_name = getattr(component, "name", component.__class__.__name__)
    raise ConfigurationError(
        f"Component '{type_name}' does not implement to_dict() and cannot be "
        f"serialized. Implement to_dict() returning {{'type': ..., 'params': ...}} "
        f"to make it round-trip through export_config()/from_dict()."
    )


def _instantiate(cfg: dict[str, Any], _stage_hint: str = "") -> Any:
    """Instantiate a component from a ``{type, params}`` dict using the registry."""
    type_name = cfg.get("type", "")
    params: dict[str, Any] = cfg.get("params", {})

    if not type_name:
        raise ConfigurationError(f"Component configuration must include a 'type' key: {cfg}")

    cls = _COMPONENT_REGISTRY.get(type_name)
    if cls is None:
        raise ConfigurationError(
            f"Unknown component type '{type_name}'. Known types: {list(_COMPONENT_REGISTRY)}"
        )

    try:
        return cls(**params)
    except TypeError as exc:
        raise ConfigurationError(
            f"Failed to instantiate '{type_name}' with params {params}: {exc}"
        ) from exc


# ============================================================================
# Component coercion — objects / strings / dicts → component instances
# ============================================================================

# vendor → {stage: (component_class, model_kwarg)}
_STRING_VENDORS: dict[str, dict[str, tuple[type, str]]] = {}


def _build_string_vendors() -> dict[str, dict[str, tuple[type, str]]]:
    """Build the vendor→stage shorthand table (lazy imports, optional deps guarded)."""
    from raglan.embedders.huggingface import HuggingFaceEmbedder
    from raglan.embedders.openai import OpenAIEmbedder
    from raglan.expanders.openai import OpenAIExpander

    vendors: dict[str, dict[str, tuple[type, str]]] = {
        "openai": {
            "embedder": (OpenAIEmbedder, "model"),
            "expander": (OpenAIExpander, "model"),
        },
        "huggingface": {"embedder": (HuggingFaceEmbedder, "model_name")},
    }

    try:
        from raglan.embedders.dashscope import DashScopeEmbedder

        vendors["dashscope"] = {"embedder": (DashScopeEmbedder, "model")}
    except ImportError:
        pass

    try:
        from raglan.expanders.litellm import LiteLLMExpander

        vendors["litellm"] = {"expander": (LiteLLMExpander, "model")}
    except ImportError:
        pass

    return vendors


def _string_to_component(s: str, hint: str) -> Any:
    """Resolve a string shorthand to a component instance.

    Two forms are supported:

    1. A bare registry type name — ``"bm25"``, ``"rrf"``, ``"passthrough"``.
    2. ``"vendor:model"`` — ``"openai:text-embedding-3-small"`` (embedder),
       ``"openai:gpt-4o-mini"`` (expander), ``"huggingface:BAAI/..."``.
    """
    global _STRING_VENDORS
    if not _STRING_VENDORS:
        _STRING_VENDORS = _build_string_vendors()

    # 1) Bare registry type name, e.g. "bm25", "rrf", "passthrough".
    if s in _COMPONENT_REGISTRY:
        return _COMPONENT_REGISTRY[s]()

    # 2) vendor:model shorthand.
    if ":" in s:
        vendor, model = s.split(":", 1)
        table = _STRING_VENDORS.get(vendor)
        if table is not None and hint in table:
            cls, param = table[hint]
            return cls(**{param: model})

    raise ConfigurationError(
        f"Cannot interpret '{s}' as {hint}. Pass a component object, "
        f"a {{'type': ...}} dict, or 'vendor:model'."
    )


def _coerce_component(value: Any, hint: str) -> Any:
    """Normalise a component spec into a live instance.

    Accepts an already-instantiated object, a ``"vendor:model"`` / registry
    type-name string, or a ``{"type": ..., "params": ...}`` dict.  ``None``
    passes through unchanged (meaning "use the default").
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return _instantiate(value, hint)
    if isinstance(value, str):
        return _string_to_component(value, hint)
    return value


def _is_identity_expander(obj: Any) -> bool:
    from raglan.expanders.identity import IdentityExpander

    return isinstance(obj, IdentityExpander)


def _is_passthrough_builder(obj: Any) -> bool:
    from raglan.context_builders.passthrough import PassthroughBuilder

    return isinstance(obj, PassthroughBuilder)
