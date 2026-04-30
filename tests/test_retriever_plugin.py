"""Tests for RetrieverPlugin protocol and pipeline integration."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import PipelineExecutor, RRFFusionStep
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult
from cam_rag.retrieval.plugin import PluginRegistry, RetrieverPlugin


# ---------------------------------------------------------------------------
# Real TF-IDF retriever plugin (no mocks, validates the protocol)
# ---------------------------------------------------------------------------


class TFIDFRetrieverPlugin:
    """Simple TF-IDF retriever that implements RetrieverPlugin protocol."""

    def __init__(self) -> None:
        self._documents: list[RetrievalDocument] = []
        self._tfidf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "tfidf"

    def index(self, documents: list[RetrievalDocument]) -> None:
        self._documents = list(documents)
        n = len(documents)
        if n == 0:
            return

        # Compute document frequencies
        df: Counter[str] = Counter()
        doc_terms: list[dict[str, int]] = []
        for doc in documents:
            terms = _tokenize(doc.text)
            tf = Counter(terms)
            doc_terms.append(tf)
            for term in set(terms):
                df[term] += 1

        # Compute IDF
        self._idf = {
            term: math.log((n + 1) / (freq + 1)) + 1
            for term, freq in df.items()
        }

        # Compute TF-IDF vectors
        self._tfidf = []
        for tf in doc_terms:
            total = sum(tf.values()) or 1
            tfidf = {
                term: (count / total) * self._idf.get(term, 1.0)
                for term, count in tf.items()
            }
            self._tfidf.append(tfidf)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # Score each document
        scored: list[tuple[int, float]] = []
        for i, tfidf in enumerate(self._tfidf):
            score = sum(tfidf.get(term, 0.0) for term in query_terms)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: -x[1])
        return [
            RetrievalResult(
                doc_id=self._documents[idx].doc_id,
                text=self._documents[idx].text,
                score=score,
                rank=rank + 1,
                metadata={},
            )
            for rank, (idx, score) in enumerate(scored[:k])
        ]


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9]+", text)]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(id="doc_0", document_id="doc_0", text="The cat sat on the mat"),
        Chunk(id="doc_1", document_id="doc_1", text="Machine learning optimization techniques"),
        Chunk(id="doc_2", document_id="doc_2", text="Quantum computing and qubits"),
        Chunk(id="doc_3", document_id="doc_3", text="The stock market crashed today"),
        Chunk(id="doc_4", document_id="doc_4", text="Photosynthesis in plants and algae"),
    ]


class TestRetrieverPluginProtocol:
    def test_isinstance_check(self) -> None:
        plugin = TFIDFRetrieverPlugin()
        assert isinstance(plugin, RetrieverPlugin)

    def test_plugin_name(self) -> None:
        plugin = TFIDFRetrieverPlugin()
        assert plugin.name == "tfidf"

    def test_plugin_index_and_retrieve(self) -> None:
        plugin = TFIDFRetrieverPlugin()
        docs = [
            RetrievalDocument(doc_id="a", text="cat on the mat", metadata={}),
            RetrievalDocument(doc_id="b", text="machine learning", metadata={}),
        ]
        plugin.index(docs)
        results = plugin.retrieve("cat", k=5)
        assert len(results) >= 1
        assert results[0].doc_id == "a"

    def test_empty_query_returns_nothing(self) -> None:
        plugin = TFIDFRetrieverPlugin()
        docs = [
            RetrievalDocument(doc_id="a", text="some text", metadata={}),
        ]
        plugin.index(docs)
        results = plugin.retrieve("", k=5)
        assert results == []


class TestPluginRegistry:
    def test_register_and_get(self) -> None:
        registry = PluginRegistry()
        plugin = TFIDFRetrieverPlugin()
        registry.register(plugin)
        assert registry.get("tfidf") is plugin

    def test_get_unknown_raises(self) -> None:
        registry = PluginRegistry()
        try:
            registry.get("nonexistent")
            assert False, "Should have raised"
        except KeyError:
            pass

    def test_all_plugins(self) -> None:
        registry = PluginRegistry()
        p1 = TFIDFRetrieverPlugin()
        registry.register(p1)
        assert len(registry.all) == 1
        assert registry.all[0].name == "tfidf"

    def test_names(self) -> None:
        registry = PluginRegistry()
        p1 = TFIDFRetrieverPlugin()
        registry.register(p1)
        assert registry.names == ["tfidf"]

    def test_len(self) -> None:
        registry = PluginRegistry()
        assert len(registry) == 0
        registry.register(TFIDFRetrieverPlugin())
        assert len(registry) == 1


class TestPluginPipelineIntegration:
    def test_plugin_participates_in_rrf_fusion(self) -> None:
        """Plugin results should be included in N-way RRF fusion."""
        chunks = _make_chunks()
        plugin = TFIDFRetrieverPlugin()
        spec = RAGAppSpec(
            name="test",
            use_pipeline=True,
            retriever_plugins=[plugin],
        )
        ctx = build_context(
            query_text="cat mat",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=5,
        )

        # Run sparse + dense + RRF (which includes plugin)
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert len(ctx.evidence) > 0
        # The plugin should have been indexed and used
        assert "_plugins_indexed" in ctx.extras
        assert "tfidf" in ctx.extras["_plugins_indexed"]

    def test_empty_plugin_list_no_effect(self) -> None:
        """No plugins → standard 2-way fusion, no changes."""
        chunks = _make_chunks()
        spec = RAGAppSpec(
            name="test",
            use_pipeline=True,
            retriever_plugins=[],
        )
        ctx = build_context(
            query_text="cat mat",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=5,
        )
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)
        assert len(ctx.evidence) > 0
        assert "_plugins_indexed" not in ctx.extras

    def test_plugin_with_no_results_doesnt_break(self) -> None:
        """A plugin that returns no results should not break fusion."""

        class EmptyPlugin:
            name = "empty"

            def index(self, documents: list[Any]) -> None:
                pass

            def retrieve(self, query: str, k: int = 10) -> list[Any]:
                return []

        chunks = _make_chunks()
        spec = RAGAppSpec(
            name="test",
            use_pipeline=True,
            retriever_plugins=[EmptyPlugin()],
        )
        ctx = build_context(
            query_text="cat",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=5,
        )
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)
        assert len(ctx.evidence) > 0
