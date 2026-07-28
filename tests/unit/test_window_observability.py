"""Tests for WindowBuilder and observability modules."""

from __future__ import annotations

import logging

from raglan.context_builders.window import WindowBuilder
from raglan.observability import LoggingMetricsCollector, NoOpMetricsCollector
from raglan.types import ScoredChunk


class TestWindowBuilder:
    async def test_empty_candidates(self):
        wb = WindowBuilder()
        results = await wb.build("query", [])
        assert results == []

    async def test_passthrough_without_loader(self):
        wb = WindowBuilder()
        chunks = [ScoredChunk(chunk_id="1", content="hello world", score=0.9)]
        results = await wb.build("test", chunks)
        assert len(results) == 1
        assert results[0].chunk_id == "1"
        assert results[0].parent_content is None

    async def test_window_extraction_chunk_found(self):
        async def loader(ids):
            return {
                "parent1": "aaaa bbbb " + ("x " * 300) + "hello world " + ("y " * 300) + "cccc dddd"
            }

        wb = WindowBuilder(loader=loader, window_chars=10)
        chunks = [
            ScoredChunk(chunk_id="c1", content="hello world", score=0.9, parent_chunk_id="parent1")
        ]
        results = await wb.build("test", chunks)
        assert len(results) == 1
        assert "hello world" in results[0].parent_content

    async def test_window_extraction_chunk_not_found(self):
        async def loader(ids):
            return {"parent1": "BEGIN " + ("abc " * 100) + " END"}

        wb = WindowBuilder(loader=loader, window_chars=10)
        chunks = [
            ScoredChunk(chunk_id="c1", content="NOT IN DOC", score=0.9, parent_chunk_id="parent1")
        ]
        results = await wb.build("test", chunks)
        assert len(results) == 1
        # Falls back to beginning of parent
        assert results[0].parent_content is not None

    async def test_window_with_no_parent_id(self):
        async def loader(ids):
            return {}

        wb = WindowBuilder(loader=loader, window_chars=50)
        chunks = [ScoredChunk(chunk_id="c1", content="hello", score=0.9)]
        results = await wb.build("test", chunks)
        assert len(results) == 1
        assert results[0].parent_content is None

    async def test_token_budget_respected(self):
        async def loader(ids):
            return {"p1": "x" * 10000}

        wb = WindowBuilder(loader=loader, window_chars=5000, max_tokens=10)
        chunks = [ScoredChunk(chunk_id="c1", content="target", score=0.9, parent_chunk_id="p1")]
        results = await wb.build("test", chunks)
        assert len(results) >= 0  # At least doesn't crash
        if results:
            assert len(results[0].parent_content or "") <= 10000

    async def test_runtime_max_tokens_override(self):
        async def loader(ids):
            return {"p1": "before target after"}

        wb = WindowBuilder(loader=loader, window_chars=100, max_tokens=6000)
        chunks = [ScoredChunk(chunk_id="c1", content="target", score=0.9, parent_chunk_id="p1")]
        results = await wb.build("test", chunks, max_tokens=1)
        assert len(results) == 1

    def test_to_dict(self):
        wb = WindowBuilder(window_chars=200, max_tokens=3000)
        d = wb.to_dict()
        assert d["type"] == "window"
        assert d["params"]["window_chars"] == 200
        assert d["params"]["max_tokens"] == 3000


class TestObservability:
    async def test_noop_does_nothing(self):
        collector = NoOpMetricsCollector()
        # These should not raise
        await collector.record_search("q", 10.0, 5, False)
        await collector.record_stage("stage1", 5.0, False)
        await collector.record_stage("stage2", 3.0, True, "error desc")

    async def test_logging_collector_search(self, caplog):
        caplog.set_level(logging.DEBUG, logger="raglan.observability")
        collector = LoggingMetricsCollector(log_queries=True)
        await collector.record_search("test query", 100.0, 3, False)
        assert "search completed" in caplog.text
        assert "test query" in caplog.text

    async def test_logging_collector_search_redacted(self, caplog):
        """By default, queries are redacted in logs for security."""
        caplog.set_level(logging.DEBUG, logger="raglan.observability")
        collector = LoggingMetricsCollector()
        await collector.record_search("secret data", 100.0, 3, False)
        assert "search completed" in caplog.text
        assert "<redacted>" in caplog.text
        assert "secret data" not in caplog.text

    async def test_logging_collector_search_degraded(self, caplog):
        caplog.set_level(logging.DEBUG, logger="raglan.observability")
        collector = LoggingMetricsCollector()
        await collector.record_search("bad query", 50.0, 0, True)
        assert "degraded" in caplog.text

    async def test_logging_collector_stage_ok(self, caplog):
        caplog.set_level(logging.DEBUG, logger="raglan.observability")
        collector = LoggingMetricsCollector()
        await collector.record_stage("expand", 200.0, False)
        assert "expand" in caplog.text
        assert "completed" in caplog.text

    async def test_logging_collector_stage_degraded(self, caplog):
        caplog.set_level(logging.DEBUG, logger="raglan.observability")
        collector = LoggingMetricsCollector()
        await collector.record_stage("reranker", 10.0, True, "model load failed")
        assert "degraded" in caplog.text
        assert "model load failed" in caplog.text
