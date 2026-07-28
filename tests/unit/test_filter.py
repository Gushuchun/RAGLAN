"""Tests for the Filter system."""

from __future__ import annotations

import pytest

from raglan.types import Filter, Op


def test_simple_eq():
    f = Filter.eq("status", "published")
    assert f.field == "status"
    assert f.op == Op.EQ
    assert f.value == "published"
    assert f.children is None


def test_and_combination():
    f = Filter.all(
        Filter.eq("status", "active"),
        Filter.gte("score", 0.5),
    )
    assert f.op == Op.AND
    assert len(f.children) == 2


def test_or_combination():
    f = Filter.any(
        Filter.eq("category", "returns"),
        Filter.eq("category", "refunds"),
    )
    assert f.op == Op.OR


def test_operator_overload():
    f = Filter.eq("status", "published") & Filter.gte("score", 0.5)
    assert f.op == Op.AND
    assert len(f.children) == 2


def test_operator_overload_or():
    f = Filter.eq("a", "1") | Filter.eq("b", "2")
    assert f.op == Op.OR


def test_nested_combination():
    f = Filter.all(
        Filter.eq("status", "active"),
        Filter.any(
            Filter.eq("category", "returns"),
            Filter.gte("score", 0.8),
        ),
    )
    assert f.op == Op.AND
    assert f.children[1].op == Op.OR


def test_bool_forbidden():
    """Filter objects cannot be used in boolean contexts."""
    f = Filter.eq("status", "published")
    with pytest.raises(TypeError):
        if f:  # type: ignore[truthy-function]
            pass


def test_ne_filter():
    assert Filter.ne("status", "deleted").op == Op.NE


def test_gt_filter():
    assert Filter.gt("score", 0.5).op == Op.GT


def test_lt_filter():
    assert Filter.lt("age", 30).op == Op.LT


def test_lte_filter():
    assert Filter.lte("price", 100).op == Op.LTE


def test_in_filter():
    f = Filter.in_("category", ["a", "b"])
    assert f.op == Op.IN
    assert f.value == ["a", "b"]


def test_exists_filter():
    f = Filter.exists("metadata")
    assert f.op == Op.EXISTS
    assert f.value is None


def test_contains_filter():
    f = Filter.contains("title", "warranty")
    assert f.op == Op.CONTAINS
    assert f.value == "warranty"


def test_filter_repr():
    """Filter __repr__ is human-readable."""
    f = Filter.eq("status", "active") & Filter.gte("score", 0.5)
    r = repr(f)
    assert "status" in r
    assert "score" in r
