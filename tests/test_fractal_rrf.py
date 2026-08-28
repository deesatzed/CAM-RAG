"""
Tests for Reciprocal Rank Fusion retrieval (Phase 3A-3).

RRF combines rankings from multiple fractal levels using rank position
(1/(k + rank)) instead of raw cosine scores, making it more robust to
scale differences between levels.
"""

import numpy as np
import pytest
from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding


# Multi-paragraph doc for richer fractal structure
MULTI_PARA_TEXT = (
    "Artificial intelligence is transforming healthcare delivery worldwide. "
    "Machine learning algorithms predict patient outcomes with high accuracy.\n\n"
    "Natural language processing extracts structured insights from clinical notes. "
    "Named entity recognition identifies medications and diagnoses automatically.\n\n"
    "Computer vision systems detect anomalies in radiology scans reliably. "
    "Deep learning models outperform traditional image analysis methods."
)

SINGLE_PARA_TEXT = (
    "Robotic surgery platforms improve surgical precision during operations. "
    "Teleoperation allows remote procedures across continental distances. "
    "Haptic feedback provides tactile information to the operating surgeon. "
    "Augmented reality overlays assist with anatomical navigation during surgery."
)


@pytest.fixture
def rag_with_docs():
    """Build a FractalRAG instance with multiple documents."""
    rag = FractalRAG(backend=HashEmbedding(dim=64))
    rag.add_document("d1", MULTI_PARA_TEXT, metadata={"domain": "radiology"})
    rag.add_document("d2", SINGLE_PARA_TEXT, metadata={"domain": "surgery"})
    rag.add_document(
        "d3",
        "Pharmacogenomics uses genetic information to guide medication selection. "
        "DNA variants affect drug metabolism through cytochrome P450 enzymes. "
        "Therapeutic drug monitoring optimizes dosing for individual patients. "
        "Adverse drug reactions can be predicted using genomic biomarkers.",
        metadata={"domain": "pharmacology"},
    )
    return rag


class TestRRFBasic:
    def test_rrf_returns_standard_format(self, rag_with_docs):
        """RRF should return (results_dict, query_type) like other methods."""
        results, qtype = rag_with_docs.retrieve_rrf("AI in healthcare")
        assert isinstance(results, dict)
        assert isinstance(qtype, str)
        assert qtype in ("specification", "summary", "logic", "synthesis")

    def test_rrf_results_have_all_requested_levels(self, rag_with_docs):
        """Results dict should have keys for all requested levels."""
        results, _ = rag_with_docs.retrieve_rrf("AI in healthcare", levels=[2, 1, 0])
        assert 2 in results
        assert 1 in results
        assert 0 in results

    def test_rrf_scores_are_positive(self, rag_with_docs):
        """All RRF scores should be positive (1/(k+rank) > 0)."""
        results, _ = rag_with_docs.retrieve_rrf("AI in healthcare")
        for lvl, items in results.items():
            for entry, score in items:
                assert score > 0, f"RRF score should be positive, got {score}"

    def test_rrf_scores_are_ordered(self, rag_with_docs):
        """Results within each level should be ordered by RRF score descending."""
        results, _ = rag_with_docs.retrieve_rrf("AI in healthcare")
        for lvl, items in results.items():
            scores = [s for _, s in items]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"RRF scores not ordered at level {lvl}: {scores}"
                )

    def test_rrf_respects_k(self, rag_with_docs):
        """Should not return more results than k."""
        results, _ = rag_with_docs.retrieve_rrf("AI in healthcare", k=2)
        # Each level should have at most k entries
        for lvl, items in results.items():
            assert len(items) <= 2


class TestRRFSingleLevel:
    def test_rrf_single_level_equivalent_to_flat_ranking(self, rag_with_docs):
        """RRF with only level 2 should produce same doc ordering as flat."""
        rrf_results, _ = rag_with_docs.retrieve_rrf(
            "AI in healthcare", levels=[2], use_derivatives=False
        )
        flat_results, _ = rag_with_docs.retrieve_flat("AI in healthcare", k=10)

        rrf_docs = [e.id for e, _ in rrf_results[2]]
        flat_docs = [e.id for e, _ in flat_results[2]]
        assert rrf_docs == flat_docs, (
            f"Single-level RRF should match flat ranking: {rrf_docs} vs {flat_docs}"
        )


class TestRRFMultiLevel:
    def test_rrf_fusion_of_3_levels(self, rag_with_docs):
        """RRF with 3 levels should produce results."""
        results, _ = rag_with_docs.retrieve_rrf("AI in healthcare", levels=[2, 1, 0])
        # Should have at least one result at doc level
        assert len(results[2]) > 0

    def test_rrf_k_parameter_affects_scores(self, rag_with_docs):
        """Different rrf_k values should produce different score distributions."""
        results_60, _ = rag_with_docs.retrieve_rrf("AI in healthcare", rrf_k=60)
        results_10, _ = rag_with_docs.retrieve_rrf("AI in healthcare", rrf_k=10)

        # Same ordering, but different absolute scores
        scores_60 = [s for _, s in results_60[2]]
        scores_10 = [s for _, s in results_10[2]]
        if scores_60 and scores_10:
            # Lower rrf_k gives higher absolute scores
            assert scores_10[0] > scores_60[0], (
                "Lower rrf_k should give higher RRF scores"
            )


class TestRRFMetadata:
    def test_rrf_with_metadata_filters(self, rag_with_docs):
        """RRF should respect metadata filters."""
        results, _ = rag_with_docs.retrieve_rrf(
            "AI in healthcare",
            metadata_filters={"domain": "radiology"},
        )
        # Only d1 has domain=radiology
        doc_ids = set()
        for lvl, items in results.items():
            for entry, _ in items:
                doc_id = entry.parent if entry.parent else entry.id
                doc_ids.add(doc_id)
        assert doc_ids == {"d1"}, f"Filter should limit to d1, got {doc_ids}"

    def test_rrf_with_metadata_boost(self, rag_with_docs):
        """RRF should add metadata boost to scores."""
        results_no_boost, _ = rag_with_docs.retrieve_rrf("AI in healthcare")
        results_with_boost, _ = rag_with_docs.retrieve_rrf(
            "AI in healthcare",
            metadata_boost={
                "domain_boost": 0.1,
                "domain_target": "radiology",
            },
        )

        # d1 (radiology) should get a score boost
        def get_doc_score(results, doc_id):
            for entry, score in results.get(2, []):
                if entry.id == doc_id:
                    return score
            return None

        score_no_boost = get_doc_score(results_no_boost, "d1")
        score_with_boost = get_doc_score(results_with_boost, "d1")
        if score_no_boost is not None and score_with_boost is not None:
            assert score_with_boost > score_no_boost


class TestRRFAdaptiveIntegration:
    def test_adaptive_with_use_rrf_true(self, rag_with_docs):
        """retrieve_adaptive(use_rrf=True) should use RRF for non-spec queries."""
        results, qtype = rag_with_docs.retrieve_adaptive(
            "Summarize the main findings across all studies",
            use_rrf=True,
        )
        # Should still return valid results
        assert isinstance(results, dict)
        assert qtype in ("summary", "synthesis", "logic")

    def test_adaptive_rrf_spec_still_uses_flat(self, rag_with_docs):
        """Spec queries should still use flat even with use_rrf=True."""
        results, qtype = rag_with_docs.retrieve_adaptive(
            "What specific type of imaging does the study use?",
            use_rrf=True,
        )
        # Specification queries bypass RRF and use flat
        assert qtype == "specification"
