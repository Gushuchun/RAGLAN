"""Regression tests — prevent fixed edge cases from recurring.

Naming convention: each test should reference the bug / issue that prompted it.
"""

from __future__ import annotations

import pytest


class TestEmptyQueryRegression:
    """Edge cases around empty or blank queries."""

    @pytest.mark.asyncio
    async def test_whitespace_only_query_raises(self):
        """Regression: whitespace-only query should raise ValueError.

        A query consisting entirely of spaces was previously passed through,
        causing downstream stages to process meaningless input.
        """
        from raglan import Raglan
        from raglan.retrievers import BM25Retriever

        bm = BM25Retriever()
        await bm.index(_batch_gen([("d1", "hello", None)]))
        rag = Raglan.builder().with_retrievers([bm]).build()

        with pytest.raises(ValueError, match="non-empty"):
            await rag.search("   ")

    @pytest.mark.asyncio
    async def test_newline_only_query_raises(self):
        """\\n and \\t should also be rejected."""
        from raglan import Raglan
        from raglan.retrievers import BM25Retriever

        bm = BM25Retriever()
        await bm.index(_batch_gen([("d1", "hello", None)]))
        rag = Raglan.builder().with_retrievers([bm]).build()

        for bad in ["\n\n", "\t\t", "\n \t"]:
            with pytest.raises(ValueError):
                await rag.search(bad)


class TestChunkIdCollisionRegression:
    """Duplicate chunk IDs should not cause silent data loss."""

    @pytest.mark.asyncio
    async def test_duplicate_chunk_id_does_not_crash(self):
        """Adding a chunk with an existing ID should overwrite cleanly."""
        from raglan.retrievers import BM25Retriever

        bm = BM25Retriever()
        await bm.index(_batch_gen([("dup", "first version", None)]))
        await bm.add([("dup", "second version", None)])

        results = await bm.retrieve(["second version"], [], top_k=3)
        assert len(results[0]) >= 1
        # The updated content should be searchable
        contents = [c.content for c in results[0]]
        assert any("second version" in c for c in contents)


class TestFilterEdgeCases:
    """Edge cases around the Filter system."""

    def test_filter_with_special_characters_in_value(self):
        from raglan.types import Filter

        f = Filter.eq("path", "/usr/local/bin")
        assert f.value == "/usr/local/bin"

        f2 = Filter.eq("name", "O'Brien")
        assert f2.value == "O'Brien"

    def test_empty_all_filter(self):
        from raglan.types import Filter

        f = Filter.all()
        assert f.children == []

    def test_empty_any_filter(self):
        from raglan.types import Filter

        f = Filter.any()
        assert f.children == []


# -- helpers -------------------------------------------------------------------


async def _batch_gen(items):
    yield [(cid, content, meta) for cid, content, meta in items]
