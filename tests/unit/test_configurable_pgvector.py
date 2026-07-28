"""Tests for ConfigurablePgvectorRetriever — SQL generation and filter translation."""

from __future__ import annotations

import pytest

from raglan.retrievers.configurable_pgvector import ConfigurablePgvectorRetriever
from raglan.types import Filter


def test_default_column_mappings():
    r = ConfigurablePgvectorRetriever(
        table="public.chunks",
        id_column="chunk_id",
        content_column="body",
        embedding_column="vec",
        parent_id_column="doc_id",
        metadata_column="meta",
    )
    assert r.name == "pgvector"
    assert r.requires_embeddings is True


def test_distance_metric_validation():
    with pytest.raises(ValueError, match="distance_metric"):
        ConfigurablePgvectorRetriever(table="t", distance_metric="euclidean")


def test_build_filter_eq():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, params = r._build_filter([Filter.eq("status", "published")])
    assert "meta->>'status'" in clause
    assert "= $3" in clause
    assert params == ["published"]


def test_build_filter_and():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, params = r._build_filter(
        [
            Filter.eq("status", "published") & Filter.gte("score", 0.5),
        ]
    )
    assert "AND" in clause
    assert len(params) == 2


def test_build_filter_or():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, params = r._build_filter(
        [
            Filter.eq("cat", "a") | Filter.eq("cat", "b"),
        ]
    )
    assert "OR" in clause
    assert len(params) == 2


def test_build_filter_in():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, _params = r._build_filter(
        [
            Filter.in_("status", ["a", "b"]),
        ]
    )
    assert "ANY" in clause


def test_build_filter_exists():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, params = r._build_filter([Filter.exists("metadata_field")])
    assert "IS NOT NULL" in clause
    assert params == []


def test_build_filter_contains():
    r = ConfigurablePgvectorRetriever(table="t", metadata_column="meta")
    clause, params = r._build_filter([Filter.contains("title", "warranty")])
    assert "LIKE" in clause
    assert "%warranty%" in params[0]


def test_build_filter_empty_returns_true():
    r = ConfigurablePgvectorRetriever(table="t")
    clause, params = r._build_filter([])
    assert clause == "TRUE"
    assert params == []


def test_build_filter_no_metadata_column():
    """Without a metadata column, filters are silently ignored."""
    r = ConfigurablePgvectorRetriever(table="t", metadata_column=None)
    clause, params = r._build_filter([Filter.eq("status", "published")])
    assert clause == "TRUE"
    assert params == []


def test_load_parents_empty():
    r = ConfigurablePgvectorRetriever(table="t")
    # No parent_id_column → returns {}
    result = r._parent_id_col
    assert result is None
