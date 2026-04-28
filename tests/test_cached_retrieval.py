"""Tests for P7-02 (cached retrieval indexes) and M8-05 (adaptive params wiring)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cam_rag.query import classify_query_type, query, query_document_folder
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval import (
    DenseVectorRetriever,
    SparseBM25Retriever,
    compute_adaptive_params,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus_documents() -> list[CorpusDocument]:
    """Build a small corpus for in-memory query tests."""
    return [
        CorpusDocument(
            id="doc-triage",
            text="# Triage\n\nED triage requires immediate vital signs and acuity assignment.",
            source="triage.md",
            title="Triage",
        ),
        CorpusDocument(
            id="doc-billing",
            text="# Billing\n\nInvoices are reviewed monthly.",
            source="billing.md",
            title="Billing",
        ),
        CorpusDocument(
            id="doc-protocol",
            text="# Protocol\n\nAirway management protocol uses rapid sequence intubation.",
            source="protocol.md",
            title="Protocol",
        ),
    ]


def _write_docs(tmp_path: Path) -> None:
    """Write small markdown corpus to disk for query_document_folder tests."""
    (tmp_path / "triage.md").write_text(
        "# Triage\n\nED triage requires immediate vital signs and acuity assignment.",
        encoding="utf-8",
    )
    (tmp_path / "billing.md").write_text(
        "# Billing\n\nInvoices are reviewed monthly.",
        encoding="utf-8",
    )
    (tmp_path / "protocol.md").write_text(
        "# Protocol\n\nAirway management protocol uses rapid sequence intubation.",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# P7-02: Retriever caching -- indexes are built ONCE per query
# ---------------------------------------------------------------------------

class TestRetrieverCaching:
    """Verify that SparseBM25Retriever and DenseVectorRetriever are each
    instantiated exactly once per query, even when query expansion fires."""

    def test_retrievers_built_once_without_expansion(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="cache-test",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with (
            patch(
                "cam_rag.query.SparseBM25Retriever",
                wraps=SparseBM25Retriever,
            ) as sparse_cls,
            patch(
                "cam_rag.query.DenseVectorRetriever",
                wraps=DenseVectorRetriever,
            ) as dense_cls,
        ):
            answer = query_document_folder(tmp_path, "ED triage vital signs", spec)

        assert sparse_cls.call_count == 1, (
            f"SparseBM25Retriever built {sparse_cls.call_count} times, expected 1"
        )
        assert dense_cls.call_count == 1, (
            f"DenseVectorRetriever built {dense_cls.call_count} times, expected 1"
        )
        assert answer.evidence

    def test_retrievers_built_once_with_expansion(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="cache-expansion-test",
            supported_extensions=(".md",),
            query_expansion_enabled=True,
            expansion_terms=3,
        )

        with (
            patch(
                "cam_rag.query.SparseBM25Retriever",
                wraps=SparseBM25Retriever,
            ) as sparse_cls,
            patch(
                "cam_rag.query.DenseVectorRetriever",
                wraps=DenseVectorRetriever,
            ) as dense_cls,
        ):
            answer = query_document_folder(tmp_path, "ED triage vital signs", spec)

        assert sparse_cls.call_count == 1, (
            f"SparseBM25Retriever built {sparse_cls.call_count} times with expansion, expected 1"
        )
        assert dense_cls.call_count == 1, (
            f"DenseVectorRetriever built {dense_cls.call_count} times with expansion, expected 1"
        )
        assert answer.evidence

    def test_retrievers_built_once_via_query_api(self):
        """The in-memory ``query()`` entry point also reuses cached retrievers."""
        docs = _make_corpus_documents()
        spec = RAGAppSpec(
            name="cache-query-api",
            supported_extensions=(".md",),
            query_expansion_enabled=True,
            expansion_terms=2,
        )

        with (
            patch(
                "cam_rag.query.SparseBM25Retriever",
                wraps=SparseBM25Retriever,
            ) as sparse_cls,
            patch(
                "cam_rag.query.DenseVectorRetriever",
                wraps=DenseVectorRetriever,
            ) as dense_cls,
        ):
            answer = query("ED triage vital signs", docs, spec)

        assert sparse_cls.call_count == 1
        assert dense_cls.call_count == 1
        assert answer.evidence


# ---------------------------------------------------------------------------
# M8-05: classify_query_type heuristic
# ---------------------------------------------------------------------------

class TestClassifyQueryType:

    @pytest.mark.parametrize(
        ("query_text", "expected"),
        [
            ("What changed? Why does it matter?", "synthesis"),
            ("compare aspirin versus ibuprofen", "synthesis"),
            ("synthesize the triage protocols", "synthesis"),
            ("What is rapid sequence intubation?", "specification"),
            ("define triage acuity", "specification"),
            ("definition of airway management", "specification"),
            ("Why does triage require vital signs?", "logic"),
            ("How should we intubate?", "logic"),
            ("explain the billing cycle", "logic"),
            ("because of the protocol changes", "logic"),
            ("ED triage vital signs", "summary"),
            ("billing invoices monthly review", "summary"),
        ],
    )
    def test_classify_query_type(self, query_text: str, expected: str):
        assert classify_query_type(query_text) == expected


# ---------------------------------------------------------------------------
# M8-05: Adaptive params are wired into the query pipeline
# ---------------------------------------------------------------------------

class TestAdaptiveParamsWiring:

    def test_specification_query_uses_specification_params(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-spec",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with patch(
            "cam_rag.query.compute_adaptive_params",
            wraps=compute_adaptive_params,
        ) as cap:
            answer = query_document_folder(
                tmp_path, "What is rapid sequence intubation?", spec,
            )

        cap.assert_called_once()
        call_args = cap.call_args
        assert call_args[0][0] == "specification", (
            f"Expected query type 'specification', got '{call_args[0][0]}'"
        )
        assert answer.evidence

    def test_synthesis_query_uses_synthesis_params(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-synth",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with patch(
            "cam_rag.query.compute_adaptive_params",
            wraps=compute_adaptive_params,
        ) as cap:
            query_document_folder(
                tmp_path, "What changed? Why does it matter?", spec,
            )

        cap.assert_called_once()
        assert cap.call_args[0][0] == "synthesis"

    def test_logic_query_uses_logic_params(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-logic",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with patch(
            "cam_rag.query.compute_adaptive_params",
            wraps=compute_adaptive_params,
        ) as cap:
            query_document_folder(
                tmp_path, "Why does triage require vital signs?", spec,
            )

        cap.assert_called_once()
        assert cap.call_args[0][0] == "logic"

    def test_summary_query_uses_summary_params(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-summary",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with patch(
            "cam_rag.query.compute_adaptive_params",
            wraps=compute_adaptive_params,
        ) as cap:
            query_document_folder(
                tmp_path, "ED triage vital signs", spec,
            )

        cap.assert_called_once()
        assert cap.call_args[0][0] == "summary"

    def test_adaptive_params_applied_to_retrieval(self, tmp_path: Path):
        """Verify that the adaptive params returned by compute_adaptive_params
        are actually used to configure the retriever k values and fusion weights
        rather than the spec defaults."""
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-applied",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
            dense_weight=0.6,
            sparse_weight=0.4,
        )

        captured_calls: list[dict] = []
        real_rrf_fuse = __import__(
            "cam_rag.retrieval.fusion", fromlist=["rrf_fuse"]
        ).rrf_fuse

        def tracking_rrf_fuse(*args, **kwargs):
            captured_calls.append(kwargs)
            return real_rrf_fuse(*args, **kwargs)

        with patch("cam_rag.query.rrf_fuse", side_effect=tracking_rrf_fuse):
            query_document_folder(
                tmp_path, "What is rapid sequence intubation?", spec,
            )

        assert len(captured_calls) == 1
        call_kwargs = captured_calls[0]
        # "specification" type has dense_weight=0.7 and sparse_weight=0.3,
        # which differ from the spec defaults of 0.6/0.4.
        assert call_kwargs["weights"]["dense"] == pytest.approx(0.7, abs=0.01)
        assert call_kwargs["weights"]["sparse"] == pytest.approx(0.3, abs=0.01)

    def test_adaptive_corpus_size_passed(self, tmp_path: Path):
        """Verify corpus_size (number of chunks) is passed to compute_adaptive_params."""
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="adaptive-corpus-size",
            supported_extensions=(".md",),
            query_expansion_enabled=False,
        )

        with patch(
            "cam_rag.query.compute_adaptive_params",
            wraps=compute_adaptive_params,
        ) as cap:
            query_document_folder(tmp_path, "ED triage", spec)

        cap.assert_called_once()
        _, kwargs = cap.call_args
        assert kwargs["corpus_size"] > 0


# ---------------------------------------------------------------------------
# Regression: existing pipeline behavior preserved
# ---------------------------------------------------------------------------

class TestRegressionExistingBehavior:

    def test_query_document_folder_returns_cited_evidence(self, tmp_path: Path):
        """Mirrors test_query_pipeline.py -- confirms nothing broke."""
        (tmp_path / "triage.md").write_text(
            "# Triage\n\nED triage requires immediate vital signs and acuity assignment.",
            encoding="utf-8",
        )
        (tmp_path / "billing.md").write_text(
            "# Billing\n\nInvoices are reviewed monthly.",
            encoding="utf-8",
        )

        answer = query_document_folder(
            tmp_path,
            "ED triage vital signs",
            RAGAppSpec(name="docs", supported_extensions=(".md",)),
        )

        assert answer.grounded is True
        assert answer.citations
        assert answer.citations[0].source == "triage.md"
        assert answer.evidence[0].retriever == "hybrid_rrf"
        assert "vital" in answer.evidence[0].signals["matched_terms"]
        assert "sparse" in answer.evidence[0].signals["source_ranks"]
        assert answer.confidence > 0
        assert answer.trace.confidence_details["grounding"]["grounded"] is True

    def test_query_document_folder_reports_no_evidence(self, tmp_path: Path):
        (tmp_path / "billing.md").write_text(
            "# Billing\n\nInvoices are reviewed monthly.",
            encoding="utf-8",
        )

        answer = query_document_folder(
            tmp_path,
            "trauma airway protocol",
            RAGAppSpec(name="docs", supported_extensions=(".md",)),
        )

        assert answer.grounded is False
        assert answer.confidence == 0.0
        assert answer.citations == []
        assert "No cited evidence" in answer.answer

    def test_query_expansion_trace_recorded(self, tmp_path: Path):
        _write_docs(tmp_path)
        spec = RAGAppSpec(
            name="exp-trace",
            supported_extensions=(".md",),
            query_expansion_enabled=True,
            expansion_terms=2,
        )

        answer = query_document_folder(tmp_path, "ED triage vital signs", spec)

        # Whether expansion fires or not, the pipeline should complete cleanly
        assert answer.evidence
        assert answer.trace.retrieval_stats["expanded_query"]

    def test_in_memory_query_api_works(self):
        docs = _make_corpus_documents()
        spec = RAGAppSpec(name="inmem", supported_extensions=(".md",))

        answer = query("ED triage vital signs", docs, spec)

        assert answer.evidence
        assert answer.citations
