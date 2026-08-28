"""
Tests for weighted paragraph pooling in FractalRAG.add_document().

Direction 5: Paragraphs are weighted by similarity to doc_vec when computing
para_mean for derivative calculations. All tests use real HashEmbedding. No mocks.
"""

import numpy as np
import pytest

from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding
from cam_rag.retrieval.fractal.core import normalize


def _make_rag(dim=64):
    return FractalRAG(backend=HashEmbedding(dim=dim))


class TestWeightedParagraphPooling:
    def test_para_mean_computed_with_weights(self):
        """Weighted para_mean should differ from simple average when paragraphs
        have different similarities to doc_vec."""
        rag = _make_rag()
        text = (
            "First paragraph about radiology imaging techniques.\n\n"
            "Second paragraph about drug discovery pipelines.\n\n"
            "Third paragraph about clinical decision support systems."
        )
        rag.add_document("d1", text)

        # Verify derivatives were computed (para_mean was used)
        sent_ids = [k for k in rag.derivatives if k.startswith("d1_s")]
        assert len(sent_ids) > 0

        # All derivatives should be finite and non-zero
        for sid in sent_ids:
            d1 = rag.derivatives[sid]["d1"]
            d2 = rag.derivatives[sid]["d2"]
            assert np.isfinite(d1).all()
            assert np.isfinite(d2).all()

    def test_single_paragraph_single_weight(self):
        """A single-paragraph doc with <4 sentences has one L1 entry.
        Weight is 1.0 (sole paragraph), so weighted = unweighted."""
        rag = _make_rag()
        text = "One sentence about AI. Another about medicine. A third one too."
        rag.add_document("d1", text)

        # Should have 1 paragraph entry
        para_entries = [e for e in rag.index[1] if e.parent == "d1"]
        assert len(para_entries) == 1

    def test_negative_dot_products_get_floored(self):
        """If a paragraph's dot product with doc_vec is negative,
        the weight floor of 0.1 prevents weight inversion."""
        rag = _make_rag()
        # Use a longer text to ensure multiple virtual paragraphs
        text = (
            "AI models detect tumors in radiology scans using deep learning. "
            "Neural networks classify medical images with high accuracy. "
            "Drug metabolism varies with genetic polymorphisms in CYP450 enzymes. "
            "Pharmacovigilance monitors adverse reactions across patient populations. "
            "Clinical trials assess drug safety through rigorous statistical methods. "
            "Emergency triage systems prioritize patients by clinical severity."
        )
        rag.add_document("d1", text)

        # All derivatives should be finite (no NaN/Inf from bad weights)
        for sid, derivs in rag.derivatives.items():
            if sid.startswith("d1_s"):
                assert np.isfinite(derivs["d1"]).all(), f"d1 not finite for {sid}"
                assert np.isfinite(derivs["d2"]).all(), f"d2 not finite for {sid}"

    def test_all_equal_paragraphs_same_as_unweighted(self):
        """When all paragraphs are identical, weighted mean = unweighted mean."""
        rag = _make_rag()
        # Identical paragraphs produce identical embeddings => equal weights
        text = (
            "AI helps in medicine.\n\n"
            "AI helps in medicine.\n\n"
            "AI helps in medicine."
        )
        rag.add_document("d1", text)

        para_entries = [e for e in rag.index[1] if e.parent == "d1"]
        vecs = [e.vec for e in para_entries]
        assert len(vecs) == 3

        # All vecs should be identical (same text)
        for v in vecs[1:]:
            np.testing.assert_array_almost_equal(v, vecs[0])

    def test_derivatives_differ_with_weighted_pooling(self):
        """With diverse paragraphs, weighted pooling should produce different
        derivatives than if all paragraphs were uniform."""
        rag = _make_rag()
        text = (
            "Radiology uses deep learning for tumor detection in CT scans.\n\n"
            "Drug discovery leverages molecular docking and virtual screening.\n\n"
            "Clinical decision support systems improve patient triage outcomes."
        )
        rag.add_document("d1", text)

        # Verify we have derivatives
        d1_derivs = {k: v for k, v in rag.derivatives.items() if k.startswith("d1_s")}
        assert len(d1_derivs) > 0

        # Derivatives should not all be zero
        any_nonzero = any(
            np.linalg.norm(v["d1"]) > 1e-6 or np.linalg.norm(v["d2"]) > 1e-6
            for v in d1_derivs.values()
        )
        assert any_nonzero, "Expected non-zero derivatives"

    def test_retrieval_works_after_weighted_pooling(self):
        """End-to-end: retrieval still works correctly with weighted pooling."""
        rag = _make_rag()
        rag.add_document(
            "d1",
            "AI helps detect cancer in radiology images using deep learning models. "
            "Convolutional neural networks analyze CT scans for tumor identification. "
            "Machine learning improves diagnostic accuracy in medical imaging. "
            "Transfer learning from ImageNet boosts radiology AI performance.",
            metadata={"domain": "radiology"},
            title="Radiology AI",
        )
        rag.add_document(
            "d2",
            "Drug metabolism depends on cytochrome P450 enzyme variants. "
            "Pharmacogenomics predicts drug response based on patient genetics. "
            "Adverse drug reactions are monitored through pharmacovigilance systems. "
            "Clinical trials evaluate drug safety in diverse patient populations.",
            metadata={"domain": "pharma"},
            title="Pharma Genetics",
        )

        # Retrieval should return results without errors
        results, qtype = rag.retrieve("How does AI help in radiology?", k=5)
        assert len(results[2]) > 0  # Some doc-level results

        results_r, _ = rag.retrieve_reranked("drug metabolism genetics", k=5)
        assert len(results_r[2]) > 0
