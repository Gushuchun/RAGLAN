"""Tests for the incremental-configuration and config-template API.

Covers the new construction styles:

- ``Raglan()`` empty instance + ``add_retriever`` / ``set_*`` setters
- ``Raglan([retrievers])`` direct instantiation with positional retrievers
- ``Raglan.config()`` template + ``Raglan.from_config()``
- object / ``"vendor:model"`` string / ``{"type": ...}`` dict component forms

The existing ``Raglan.builder()`` / ``Raglan.from_dict()`` / ``Raglan(pipeline)``
styles are covered by ``test_builder.py`` and ``test_raglan_coverage.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from raglan.exceptions import ConfigurationError
from raglan.raglan import Raglan, RaglanBuilder
from raglan.retrievers.bm25 import BM25Retriever

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_bm25() -> BM25Retriever:
    bm = BM25Retriever()

    async def gen():
        yield [("d1", "return policy for damaged items", None)]

    asyncio.run(bm.index(gen()))
    return bm


def _search(rag: Raglan, query: str = "return"):
    return asyncio.run(rag.search(query))


# ---------------------------------------------------------------------------
# empty instance + incremental setters
# ---------------------------------------------------------------------------


class TestEmptyInstance:
    def test_empty_instance_is_configurable(self):
        rag = Raglan()
        assert rag._builder is not None
        assert rag._pipeline is None

    def test_search_without_retriever_raises(self):
        rag = Raglan()
        with pytest.raises(ConfigurationError, match="At least one Retriever"):
            _search(rag)

    def test_add_retriever_then_search(self):
        bm = _make_bm25()
        rag = Raglan()
        rag.add_retriever(bm)
        results, _trace = _search(rag)
        assert len(results) >= 1

    def test_add_retrievers_multiple(self):
        bm1 = _make_bm25()
        bm2 = BM25Retriever()
        rag = Raglan().add_retrievers([bm1, bm2])
        assert len(rag._builder._retrievers) == 2  # type: ignore[union-attr]

    def test_setters_return_self_for_chaining(self):
        bm = _make_bm25()
        rag = Raglan()
        assert rag.add_retriever(bm) is rag
        assert rag.set_fusion("rrf") is rag
        assert rag.set_fallback_mode("degrade") is rag
        assert rag.set_trace_level("normal") is rag

    def test_chainable_construction(self):
        bm = _make_bm25()
        rag = Raglan().add_retriever(bm).set_fusion("rrf")
        results, _ = _search(rag)
        assert len(results) >= 1

    def test_setter_after_search_raises(self):
        bm = _make_bm25()
        rag = Raglan([bm])
        _search(rag)
        with pytest.raises(ConfigurationError, match="already built"):
            rag.add_retriever(bm)


# ---------------------------------------------------------------------------
# component forms: object / string / dict
# ---------------------------------------------------------------------------


class TestComponentForms:
    def test_set_embedder_string_shorthand(self):
        rag = Raglan()
        rag.set_embedder("openai:text-embedding-3-small")
        embedder = rag._builder._embedder  # type: ignore[union-attr]
        assert embedder is not None
        assert embedder._model == "text-embedding-3-small"

    def test_set_embedder_dict(self):
        rag = Raglan()
        rag.set_embedder({"type": "openai_embedder", "params": {"model": "x-small"}})
        embedder = rag._builder._embedder  # type: ignore[union-attr]
        assert embedder is not None
        assert embedder._model == "x-small"

    def test_set_fusion_string(self):
        from raglan.fusion.rrf import RRFFusion

        rag = Raglan()
        rag.set_fusion("rrf")
        assert isinstance(rag._builder._fusion, RRFFusion)  # type: ignore[union-attr]

    def test_set_expander_string_shorthand(self):
        rag = Raglan()
        rag.set_expander("openai:gpt-4o-mini")
        expander = rag._builder._expander  # type: ignore[union-attr]
        assert expander is not None
        assert expander._model == "gpt-4o-mini"

    def test_set_reranker_none_disables(self):
        rag = Raglan()
        assert rag.set_reranker(None) is rag
        assert rag._builder._reranker is None  # type: ignore[union-attr]

    def test_bad_string_raises(self):
        rag = Raglan()
        with pytest.raises(ConfigurationError, match="Cannot interpret"):
            rag.set_embedder("nonsense:bad")


# ---------------------------------------------------------------------------
# direct instantiation with positional retrievers
# ---------------------------------------------------------------------------


class TestDirectInstantiation:
    def test_single_retriever_positional(self):
        bm = _make_bm25()
        rag = Raglan(bm)
        results, _ = _search(rag)
        assert len(results) >= 1

    def test_list_retrievers_positional(self):
        bm = _make_bm25()
        rag = Raglan([bm])
        results, _ = _search(rag)
        assert len(results) >= 1

    def test_legacy_pipeline_positional(self):
        from raglan.context_builders.passthrough import PassthroughBuilder
        from raglan.expanders.identity import IdentityExpander
        from raglan.fusion.rrf import RRFFusion
        from raglan.pipeline import Pipeline

        bm = _make_bm25()
        p = Pipeline([IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()])
        rag = Raglan(p)
        assert rag._builder is None
        results, _ = _search(rag)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# config template
# ---------------------------------------------------------------------------


class TestConfigTemplate:
    def test_config_returns_template(self):
        cfg = Raglan.config()
        assert cfg["retrievers"] == []
        assert cfg["fusion"] == "rrf"
        assert cfg["fallback_mode"] == "degrade"
        assert "expander" in cfg and "embedder" in cfg and "reranker" in cfg

    def test_config_is_fresh_each_call(self):
        cfg1 = Raglan.config()
        cfg1["retrievers"].append({"type": "bm25"})
        cfg2 = Raglan.config()
        assert cfg2["retrievers"] == []

    def test_from_config_with_retrievers(self):
        bm = _make_bm25()
        # Reuse the indexed instance via direct object in the list.
        rag = Raglan.from_config({"retrievers": [bm]})
        results, _ = _search(rag)
        assert len(results) >= 1

    def test_from_config_dict_retriever(self):
        cfg = Raglan.config()
        cfg["retrievers"].append({"type": "bm25", "params": {}})
        rag = Raglan.from_config(cfg)
        # from_config builds immediately → a live pipeline, no internal builder.
        assert rag._builder is None
        assert rag._pipeline is not None
        assert len(rag._pipeline.iter_stages()) >= 2  # expander + retriever(s) + ...

    def test_from_config_ignores_none_values(self):
        bm = _make_bm25()
        cfg = Raglan.config()
        cfg["retrievers"].append(bm)
        rag = Raglan.from_config(cfg)
        # None expander/embedder/reranker in the template are skipped.
        exported = rag.export_config()
        assert "expander" not in exported
        assert "embedder" not in exported
        assert "reranker" not in exported


# ---------------------------------------------------------------------------
# builder compatibility
# ---------------------------------------------------------------------------


class TestBuilderCompat:
    def test_builder_still_returns_raglan_builder(self):
        assert isinstance(Raglan.builder(), RaglanBuilder)

    def test_builder_build_still_works(self):
        bm = _make_bm25()
        rag = Raglan.builder().with_retrievers([bm]).build()
        results, _ = _search(rag)
        assert len(results) >= 1

    def test_builder_add_retriever_appends(self):
        bm = _make_bm25()
        builder = RaglanBuilder().add_retriever(bm)
        assert builder._retrievers == [bm]


# ---------------------------------------------------------------------------
# export / round-trip
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_unbuilt_incremental(self):
        bm = _make_bm25()
        rag = Raglan([bm])
        cfg = rag.export_config()
        assert cfg["fallback_mode"] == "degrade"
        assert len(cfg["retrievers"]) == 1
        assert cfg["retrievers"][0]["type"] == "bm25"

    def test_from_dict_string_shorthand_fusion(self):
        bm = _make_bm25()
        rag = Raglan.from_dict({"retrievers": [bm], "fusion": "rrf"})
        results, _ = _search(rag)
        assert len(results) >= 1
