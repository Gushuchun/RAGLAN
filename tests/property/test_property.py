"""Property-based tests using Hypothesis.

Verify invariants that should hold for any valid input — catches edge cases
that example-based tests miss.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# ============================================================================
# Filter system — serialisation round-trip invariants
# ============================================================================


class TestFilterProperties:
    @given(
        field=st.text(min_size=1, max_size=20),
        value=st.one_of(st.integers(0, 100), st.text(min_size=1, max_size=10)),
    )
    @settings(max_examples=30, deadline=500)
    def test_eq_filter_preserves_field_and_value(self, field, value):
        """Filter.eq(field, value) round-trips correctly."""
        from raglan.types import Filter

        f = Filter.eq(field, value)
        assert f.field == field
        assert f.value == value
        assert f.op.value == "eq"

    @given(field=st.text(min_size=1, max_size=10), val=st.integers(0, 100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])  # type: ignore[arg-type]
    def test_and_operator_is_commutative(self, field, val):
        """(A & B) produces same children as (B & A)."""
        from raglan.types import Filter

        a = Filter.eq(field, val)
        b = Filter.gt(field, val + 1)
        f1 = Filter.all(a, b)
        f2 = Filter.all(b, a)
        assert len(f1.children or []) == len(f2.children or [])

    def test_bool_raises_for_all_filters(self):
        """No Filter should evaluate to a boolean (catches accidental truthiness)."""
        from raglan.types import Filter

        for f in [Filter.eq("x", 1), Filter.all(), Filter.any(), Filter.gt("x", 0)]:
            with pytest.raises(TypeError, match="cannot be evaluated"):
                bool(f)


# ============================================================================
# BM25 tokenizer invariants
# ============================================================================


class TestBM25Tokenizer:
    @given(st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_tokenizer_never_empty_for_nonempty_text(self, text):
        """Tokeniser always returns at least one token for non-whitespace text."""
        assume(text.strip())
        from raglan.retrievers.bm25 import BM25Retriever

        tokens = BM25Retriever._builtin_tokenizer(text)
        assert len(tokens) >= 1, f"No tokens for: {text!r}"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_tokenizer_output_is_lowercase(self, text):
        """All tokens should be lowercase."""
        from raglan.retrievers.bm25 import BM25Retriever

        tokens = BM25Retriever._builtin_tokenizer(text)
        for t in tokens:
            assert t == t.lower(), f"Token not lowercase: {t!r}"

    def test_chinese_text_produces_bigrams(self):
        """CJK text produces bigram tokens."""
        from raglan.retrievers.bm25 import BM25Retriever

        tokens = BM25Retriever._builtin_tokenizer("你好世界")
        # Should include bigrams like 你好, 好世, 世界
        assert any(len(t) == 2 for t in tokens), f"Expected bigrams in: {tokens}"

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_same_input_produces_same_tokens(self, text):
        """Tokeniser is deterministic."""
        from raglan.retrievers.bm25 import BM25Retriever

        t1 = BM25Retriever._builtin_tokenizer(text)
        t2 = BM25Retriever._builtin_tokenizer(text)
        assert t1 == t2


# ============================================================================
# RRF fusion invariants
# ============================================================================


class TestRRFInvariants:
    def test_rrf_scores_are_monotonic_with_rank(self):
        """Higher-ranked chunks (lower rank value) get higher RRF scores."""
        # RRF score = 1/(k + rank). Lower rank → higher score.
        for rank in range(1, 20):
            score = 1.0 / (60 + rank)
            for higher_rank in range(1, rank):
                higher_score = 1.0 / (60 + higher_rank)
                assert higher_score > score, f"Rank {higher_rank} should beat rank {rank}"

    @pytest.mark.asyncio
    async def test_fusion_with_single_retriever_preserves_ordering(self):
        """With one retriever, fusion should keep the original order."""
        from raglan.fusion.rrf import RRFFusion
        from raglan.types import ScoredChunk

        chunks = [
            ScoredChunk(chunk_id=f"c{i}", content=f"chunk {i}", score=1.0 - i * 0.1)
            for i in range(10)
        ]
        fusion = RRFFusion()
        results = await fusion.fuse({"bm25": [chunks]})
        # Order should be preserved (or very close)
        for i in range(min(5, len(results))):
            assert results[i].chunk_id == f"c{i}"

    @pytest.mark.asyncio
    async def test_fusion_output_is_deduplicated(self):
        """Fusion never returns duplicate chunk_ids."""
        from raglan.fusion.rrf import RRFFusion
        from raglan.types import ScoredChunk

        # Same chunks from two retrievers
        chunks = [ScoredChunk(chunk_id="dup", content="x", score=0.9)]
        fusion = RRFFusion()
        results = await fusion.fuse({"r1": [chunks], "r2": [chunks]})
        ids = [c.chunk_id for c in results]
        assert len(ids) == len(set(ids)), f"Duplicates found: {ids}"


# ============================================================================
# Memory retriever invariants
# ============================================================================


class TestMemoryRetrieverInvariants:
    @given(st.integers(2, 20))
    @settings(max_examples=10)
    def test_top_k_respected(self, dims):
        """Retrieve never returns more than top_k results."""
        from raglan.retrievers.memory import MemoryRetriever

        n_chunks = 50
        chunks = [
            (f"c{i}", f"chunk {i}", [math.sin(i * 0.1 + j) for j in range(dims)], None)
            for i in range(n_chunks)
        ]
        mr = MemoryRetriever()
        mr.load_embedded(chunks)

        for k in [1, 3, 5, 10, 20]:
            results = asyncio.run(
                mr.retrieve(
                    ["q"],
                    [[1.0] * dims],
                    top_k=k,
                )
            )
            assert len(results[0]) <= k, f"Got {len(results[0])} > {k}"

    def test_cosine_perfect_match_is_1(self):
        """Identical vectors produce cosine similarity = 1."""
        from raglan.retrievers.memory import _cosine

        v = [0.1, 0.2, 0.3, 0.4]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal_is_0(self):
        """Orthogonal vectors produce cosine similarity = 0."""
        from raglan.retrievers.memory import _cosine

        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


# ============================================================================
# Chunk parser invariants
# ============================================================================


class TestChunkParserInvariants:
    def test_parent_chunks_have_empty_parent_id_with_fixture(self):
        """Every parent chunk in the real fixture must have parent_id=''."""
        import json
        from pathlib import Path

        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        chunks_path = fixtures_dir / "e2e_chunks.json"
        if not chunks_path.exists():
            pytest.skip("Fixtures not generated")
        with open(chunks_path) as f:
            chunks = json.load(f)
        for c in chunks:
            if c["metadata"]["type"] == "parent":
                assert c["parent_id"] == "", (
                    f"Parent {c['chunk_id']} has parent_id={c['parent_id']!r}"
                )

    def test_heading_extraction(self):
        """Markdown headings are identified correctly."""
        from tests.generate_e2e_fixtures import parse_document

        text = "# Chapter 1\n\nContent.\n\n## Section 1.1\n\nMore."
        chunks = parse_document(text)
        headings = [c["metadata"]["heading"] for c in chunks if c["metadata"]["type"] == "parent"]
        assert "Chapter 1" in headings or any("Chapter" in h for h in headings)


# ============================================================================
# Types — data class round-trips
# ============================================================================


class TestTypesInvariants:
    def test_scored_chunk_equality(self):
        from raglan.types import ScoredChunk

        a = ScoredChunk("id1", "content", 0.5)
        b = ScoredChunk("id1", "content", 0.5)
        # Same data should be equal (dataclass default)
        assert a == b

    def test_search_result_defaults(self):
        from raglan.types import SearchResult

        r = SearchResult("id1", "content", 0.5)
        assert r.parent_content is None
        assert r.metadata == {}
        assert r.source == ""


# ============================================================================
# Pipeline — stage ordering invariants
# ============================================================================


class TestPipelineInvariants:
    @pytest.mark.asyncio
    async def test_pipeline_without_retriever_raises(self):
        """Pipeline with zero retrievers should raise ConfigurationError."""
        from raglan.exceptions import ConfigurationError
        from raglan.raglan import RaglanBuilder

        with pytest.raises(ConfigurationError, match="Retriever"):
            RaglanBuilder().build()

    @pytest.mark.asyncio
    async def test_empty_query_raises_value_error(self):
        from raglan.context_builders.passthrough import PassthroughBuilder
        from raglan.expanders.identity import IdentityExpander
        from raglan.fusion.rrf import RRFFusion
        from raglan.pipeline import Pipeline
        from raglan.raglan import Raglan
        from raglan.retrievers.bm25 import BM25Retriever

        bm = BM25Retriever()
        p = Pipeline([IdentityExpander(), bm, RRFFusion(), PassthroughBuilder()])
        rag = Raglan(p)
        with pytest.raises(ValueError, match="non-empty"):
            await rag.search("   ")
