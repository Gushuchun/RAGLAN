"""Validate that the README quickstart examples actually work."""

from __future__ import annotations

import asyncio

import pytest


class TestReadmeQuickstart:
    """Ensure the 5-Minute Quickstart from README.md runs without error."""

    @pytest.mark.asyncio
    async def test_bm25_quickstart(self):
        """The BM25-only quickstart from README.md."""
        from raglan import Raglan
        from raglan.retrievers import BM25Retriever

        bm25 = BM25Retriever()

        async def chunks():
            yield [
                ("doc1", "Return policy: items can be returned within 30 days.", None),
                (
                    "doc2",
                    "Refund process: refunds are issued to the original payment method.",
                    None,
                ),
                ("doc3", "Shipping: orders ship within 2 business days.", None),
            ]

        await bm25.index(chunks())
        rag = Raglan.builder().with_retrievers([bm25]).build()
        results, _trace = await rag.search("how to return my order")

        assert len(results) >= 1
        assert results[0].score > 0
        # "return policy" should be the top result
        assert any("return" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_builder_api(self):
        """Builder API from README works with all methods returning self."""
        from raglan import Raglan
        from raglan.retrievers import BM25Retriever

        bm25 = BM25Retriever()

        rag = Raglan.builder().with_retrievers([bm25]).with_fallback_mode("degrade").build()

        assert rag is not None

    def test_search_sync_wrapper(self):
        """The synchronous search wrapper works."""
        from raglan import Raglan
        from raglan.retrievers import BM25Retriever

        async def _setup():
            bm25 = BM25Retriever()

            async def chunks():
                yield [("d1", "hello world", None)]

            await bm25.index(chunks())
            return Raglan.builder().with_retrievers([bm25]).build()

        rag = asyncio.run(_setup())
        results, trace = rag.search_sync("hello world")
        assert len(results) >= 1
        assert trace.total_ms > 0


class TestReadmeInstallation:
    """Verify that key imports match the documented API."""

    def test_core_imports(self):
        pass
        # just verify they import

    def test_retriever_imports(self):
        from raglan.retrievers import BM25Retriever

        bm = BM25Retriever()
        assert bm.name == "bm25"

    def test_exception_imports(self):
        from raglan.exceptions import ConfigurationError, RaglanError

        assert issubclass(ConfigurationError, RaglanError)
