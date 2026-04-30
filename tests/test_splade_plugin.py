"""Tests for the SPLADE retriever plugin.

All tests use a REAL SPLADE model downloaded from HuggingFace.
No mocks, no placeholders, no simulation.

Uses ``naver/splade_v2_distil`` as a smaller model for faster test
execution while still exercising the full SPLADE encode/retrieve path.
"""

from __future__ import annotations

import pytest

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import PipelineExecutor
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult
from cam_rag.retrieval.plugin import RetrieverPlugin
from cam_rag.retrieval.splade_plugin import SPLADEPlugin

# Use a smaller distilled model for faster test downloads/inference
_TEST_MODEL = "naver/splade_v2_distil"


# ---------------------------------------------------------------------------
# Shared fixture: a single SPLADE plugin loaded once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def splade_plugin() -> SPLADEPlugin:
    """Session-scoped SPLADE plugin -- model is loaded once, reused across tests."""
    return SPLADEPlugin(model_name_or_path=_TEST_MODEL)


@pytest.fixture()
def sample_docs() -> list[RetrievalDocument]:
    """Small set of topically diverse documents for retrieval tests."""
    return [
        RetrievalDocument(
            doc_id="bio_1",
            text="Photosynthesis converts sunlight into chemical energy in plants and algae.",
            metadata={"topic": "biology"},
        ),
        RetrievalDocument(
            doc_id="cs_1",
            text="Machine learning algorithms optimise neural networks using gradient descent.",
            metadata={"topic": "computer_science"},
        ),
        RetrievalDocument(
            doc_id="fin_1",
            text="The stock market experienced a significant crash in 2008.",
            metadata={"topic": "finance"},
        ),
        RetrievalDocument(
            doc_id="bio_2",
            text="DNA replication is a fundamental process in cell division.",
            metadata={"topic": "biology"},
        ),
        RetrievalDocument(
            doc_id="cs_2",
            text="Retrieval augmented generation combines search with language model answers.",
            metadata={"topic": "computer_science"},
        ),
    ]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestSPLADEProtocol:
    """Verify the SPLADE plugin satisfies the RetrieverPlugin protocol."""

    def test_isinstance_check(self, splade_plugin: SPLADEPlugin) -> None:
        """SPLADEPlugin must pass the runtime_checkable RetrieverPlugin check."""
        assert isinstance(splade_plugin, RetrieverPlugin)

    def test_plugin_name(self, splade_plugin: SPLADEPlugin) -> None:
        """Plugin name must be 'splade'."""
        assert splade_plugin.name == "splade"


# ---------------------------------------------------------------------------
# Core retrieval functionality
# ---------------------------------------------------------------------------


class TestSPLADERetrieval:
    """Test index + retrieve with a real SPLADE model."""

    def test_index_and_retrieve_returns_results(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """Indexing documents then retrieving returns non-empty results
        with correct types."""
        splade_plugin.index(sample_docs)
        results = splade_plugin.retrieve("machine learning neural networks", k=3)

        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.doc_id, str)
            assert isinstance(result.text, str)
            assert isinstance(result.score, float)
            assert isinstance(result.rank, int)
            assert isinstance(result.metadata, dict)

    def test_results_ranked_descending(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """Results must be sorted by descending score."""
        splade_plugin.index(sample_docs)
        results = splade_plugin.retrieve("photosynthesis plants energy", k=5)

        assert len(results) > 1
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at rank {i + 1} ({scores[i]}) must be >= "
                f"score at rank {i + 2} ({scores[i + 1]})"
            )

    def test_relevant_scores_higher(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """Topically relevant documents should score higher than irrelevant ones."""
        splade_plugin.index(sample_docs)

        # Query about biology -- bio docs should outscore finance/CS docs
        results = splade_plugin.retrieve("photosynthesis plants sunlight energy", k=5)

        # The top result should be the biology document about photosynthesis
        assert results[0].doc_id == "bio_1", (
            f"Expected bio_1 as top result, got {results[0].doc_id}"
        )

        # Bio doc score should exceed finance doc score
        bio_score = results[0].score
        fin_scores = [r.score for r in results if r.doc_id == "fin_1"]
        assert fin_scores, "Finance document should appear in results"
        assert bio_score > fin_scores[0], (
            f"Biology score ({bio_score}) should exceed finance score ({fin_scores[0]})"
        )

    def test_retrieve_k_limits_results(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """retrieve(k=2) must return at most 2 results."""
        splade_plugin.index(sample_docs)
        results = splade_plugin.retrieve("machine learning", k=2)

        assert len(results) <= 2

    def test_empty_query_returns_empty(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """An empty query string should return no results."""
        splade_plugin.index(sample_docs)

        assert splade_plugin.retrieve("", k=5) == []
        assert splade_plugin.retrieve("   ", k=5) == []

    def test_ranks_are_sequential(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """Rank values must start at 1 and be sequential."""
        splade_plugin.index(sample_docs)
        results = splade_plugin.retrieve("cell division DNA", k=5)

        assert len(results) > 0
        for i, result in enumerate(results, start=1):
            assert result.rank == i, f"Expected rank {i}, got {result.rank}"

    def test_metadata_preserved(
        self,
        splade_plugin: SPLADEPlugin,
        sample_docs: list[RetrievalDocument],
    ) -> None:
        """Document metadata should be preserved in retrieval results."""
        splade_plugin.index(sample_docs)
        results = splade_plugin.retrieve("stock market crash", k=5)

        # Find the finance doc in results
        fin_results = [r for r in results if r.doc_id == "fin_1"]
        assert fin_results, "Finance doc should appear in results"
        assert fin_results[0].metadata.get("topic") == "finance"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSPLADEEdgeCases:
    """Edge cases and robustness checks."""

    def test_retrieve_before_index_returns_empty(
        self,
    ) -> None:
        """Retrieving before indexing should return an empty list."""
        plugin = SPLADEPlugin(model_name_or_path=_TEST_MODEL)
        # Force model load but skip indexing
        results = plugin.retrieve("anything", k=5)
        assert results == []

    def test_index_empty_list(
        self,
        splade_plugin: SPLADEPlugin,
    ) -> None:
        """Indexing an empty document list should not raise."""
        splade_plugin.index([])
        results = splade_plugin.retrieve("test", k=5)
        assert results == []

    def test_reindex_replaces_old_index(
        self,
        splade_plugin: SPLADEPlugin,
    ) -> None:
        """Re-indexing should replace the previous index entirely."""
        docs_a = [
            RetrievalDocument(doc_id="a", text="alpha bravo charlie", metadata={}),
        ]
        docs_b = [
            RetrievalDocument(doc_id="b", text="delta echo foxtrot", metadata={}),
        ]

        splade_plugin.index(docs_a)
        results_a = splade_plugin.retrieve("alpha bravo", k=5)
        assert any(r.doc_id == "a" for r in results_a)

        splade_plugin.index(docs_b)
        results_b = splade_plugin.retrieve("alpha bravo", k=5)
        # After re-indexing, doc "a" should no longer be found
        assert not any(r.doc_id == "a" for r in results_b)
        assert any(r.doc_id == "b" for r in results_b)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestSPLADEPipelineIntegration:
    """Verify the SPLADE plugin works inside the full pipeline."""

    def test_splade_participates_in_rrf_fusion(
        self,
        splade_plugin: SPLADEPlugin,
    ) -> None:
        """SPLADE plugin results should be included in N-way RRF fusion."""
        chunks = [
            Chunk(id="c0", document_id="c0", text="The cat sat on the mat"),
            Chunk(id="c1", document_id="c1", text="Machine learning optimization techniques"),
            Chunk(id="c2", document_id="c2", text="Quantum computing and qubits"),
            Chunk(id="c3", document_id="c3", text="The stock market crashed today"),
            Chunk(id="c4", document_id="c4", text="Photosynthesis in plants and algae"),
        ]

        spec = RAGAppSpec(
            name="splade-integration-test",
            use_pipeline=True,
            retriever_plugins=[splade_plugin],
        )
        ctx = build_context(
            query_text="machine learning",
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

        # Pipeline should have evidence
        assert len(ctx.evidence) > 0

        # The SPLADE plugin should have been indexed
        assert "_plugins_indexed" in ctx.extras
        assert "splade" in ctx.extras["_plugins_indexed"]

        # At least one RRF source should mention the plugin
        has_plugin_source = any(
            "plugin_splade" in item.signals.get("source_ranks", {})
            for item in ctx.evidence
        )
        assert has_plugin_source, (
            "Expected at least one evidence item to have plugin_splade in source_ranks"
        )
