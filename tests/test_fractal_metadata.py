"""
Tests for metadata indexing, filtering, and boosting in FractalRAG.

All tests use real HashEmbedding computations. No mocks.
"""

import numpy as np
import pytest

from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding
from cam_rag.retrieval.fractal.storage import save, load


# ============================================================
# Helpers
# ============================================================

def _make_rag(dim=64):
    return FractalRAG(backend=HashEmbedding(dim=dim))


SAMPLE_METADATA = {
    "d1": {
        "domain": "ai_diagnosis",
        "year": 2023,
        "mesh_terms": ["Artificial Intelligence", "Diagnosis"],
        "journal": "Nature Medicine",
        "title": "AI in Diagnosis",
        "authors": ["Smith", "Jones"],
    },
    "d2": {
        "domain": "ai_diagnosis",
        "year": 2020,
        "mesh_terms": ["Machine Learning", "Diagnosis"],
        "journal": "JAMA",
        "title": "ML for Disease",
        "authors": ["Brown"],
    },
    "d3": {
        "domain": "pharmacovigilance",
        "year": 2024,
        "mesh_terms": ["Drug Safety", "Adverse Effects"],
        "journal": "BMJ",
        "title": "Drug Monitoring",
        "authors": ["Lee"],
    },
    "d4": {
        "domain": "radiology",
        "year": 2019,
        "mesh_terms": ["Radiology", "Deep Learning"],
        "journal": "Nature Medicine",
        "title": "Deep Learning Radiology",
        "authors": ["Chen"],
    },
}

SAMPLE_TEXTS = {
    "d1": "Artificial intelligence helps diagnose diseases using deep learning models. Machine learning algorithms improve diagnostic accuracy significantly.",
    "d2": "Machine learning predicts disease outcomes from patient data. Classification models detect early signs of chronic conditions.",
    "d3": "Pharmacovigilance systems monitor adverse drug reactions. Safety databases track drug side effects across populations.",
    "d4": "Deep learning enhances radiology image interpretation. Neural networks detect tumors in medical imaging scans.",
}


def _build_rag_with_metadata(dim=64):
    rag = _make_rag(dim)
    for doc_id in ["d1", "d2", "d3", "d4"]:
        rag.add_document(doc_id, SAMPLE_TEXTS[doc_id], metadata=SAMPLE_METADATA[doc_id])
    return rag


# ============================================================
# Indexing tests
# ============================================================

class TestMetadataIndexing:
    def test_add_with_metadata(self):
        rag = _make_rag()
        rag.add_document("d1", "Some text.", metadata={"domain": "ai", "year": 2023})
        assert "d1" in rag.doc_metadata
        assert rag.doc_metadata["d1"]["domain"] == "ai"
        assert rag.doc_metadata["d1"]["year"] == 2023

    def test_add_without_metadata(self):
        rag = _make_rag()
        rag.add_document("d1", "Some text.")
        assert "d1" not in rag.doc_metadata

    def test_metadata_dict_preserved(self):
        meta = {"domain": "test", "year": 2025, "mesh_terms": ["A", "B"], "custom_field": 42}
        rag = _make_rag()
        rag.add_document("d1", "Some text.", metadata=meta)
        assert rag.doc_metadata["d1"] == meta

    def test_stats_count(self):
        rag = _make_rag()
        rag.add_document("d1", "Text one.", metadata={"domain": "ai"})
        rag.add_document("d2", "Text two.")
        stats = rag.stats()
        assert stats["documents_with_metadata"] == 1


# ============================================================
# Storage round-trip tests
# ============================================================

class TestMetadataStorage:
    def test_round_trip(self, tmp_path):
        rag = _build_rag_with_metadata()
        db_path = tmp_path / "test.db"
        save(db_path, rag)
        rag2 = load(db_path, HashEmbedding(dim=64))
        assert rag2.doc_metadata == rag.doc_metadata

    def test_backward_compatible_load(self, tmp_path):
        """Loading a v1 DB (no doc_metadata table) should work fine."""
        rag = _make_rag()
        rag.add_document("d1", "Some text.")
        db_path = tmp_path / "v1.db"
        save(db_path, rag)

        # Manually drop the doc_metadata table to simulate v1 schema
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("DROP TABLE IF EXISTS doc_metadata")
        conn.commit()
        conn.close()

        rag2 = load(db_path, HashEmbedding(dim=64))
        assert rag2.doc_metadata == {}
        assert len(rag2.docs) == 1

    def test_complex_metadata_round_trip(self, tmp_path):
        """Metadata with unicode and nested structures survives round-trip."""
        meta = {
            "domain": "ai_diagnosis",
            "year": 2023,
            "mesh_terms": ["Inteligencia Artificial", "Diagn\u00f3stico"],
            "authors": ["M\u00fcller", "Garc\u00eda"],
            "nested": {"key": "value"},
        }
        rag = _make_rag()
        rag.add_document("d1", "Some text.", metadata=meta)
        db_path = tmp_path / "unicode.db"
        save(db_path, rag)
        rag2 = load(db_path, HashEmbedding(dim=64))
        assert rag2.doc_metadata["d1"] == meta


# ============================================================
# Filtering tests
# ============================================================

class TestMetadataFiltering:
    def test_domain_single(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI diagnosis", k=10, metadata_filters={"domain": "pharmacovigilance"})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d3"}

    def test_domain_list(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI diagnosis", k=10, metadata_filters={"domain": ["ai_diagnosis", "radiology"]})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d1", "d2", "d4"}

    def test_year_range(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI diagnosis", k=10, metadata_filters={"year_range": (2020, 2023)})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d1", "d2"}

    def test_year_min(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI diagnosis", k=10, metadata_filters={"year_min": 2023})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d1", "d3"}

    def test_year_max(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI diagnosis", k=10, metadata_filters={"year_max": 2020})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d2", "d4"}

    def test_mesh_terms(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI", k=10, metadata_filters={"mesh_terms": ["Radiology"]})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d4"}

    def test_journal(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI", k=10, metadata_filters={"journal": "Nature Medicine"})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d1", "d4"}

    def test_combined_filters(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_flat("AI", k=10, metadata_filters={
            "domain": "ai_diagnosis",
            "year_min": 2022,
        })
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d1"}

    def test_no_metadata_no_results(self):
        """Docs without metadata should be excluded when filters are active."""
        rag = _make_rag()
        rag.add_document("d1", "AI text without metadata.")
        rag.add_document("d2", "More AI text.", metadata={"domain": "ai"})
        results, _ = rag.retrieve_flat("AI", k=10, metadata_filters={"domain": "ai"})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d2"}

    def test_filter_in_retrieve_reranked(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_reranked("AI diagnosis", k=10, metadata_filters={"domain": "pharmacovigilance"})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d3"}

    def test_filter_in_retrieve_adaptive(self):
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_adaptive("AI diagnosis", k=10, metadata_filters={"domain": "radiology"})
        doc_ids = {e.id for e, _ in results[2]}
        assert doc_ids == {"d4"}


# ============================================================
# Boosting tests
# ============================================================

class TestMetadataBoosting:
    def test_domain_boost(self):
        rag = _build_rag_with_metadata()
        # Without boost
        results_no, _ = rag.retrieve_reranked("medical imaging", k=4)
        # With domain boost for radiology
        results_yes, _ = rag.retrieve_reranked("medical imaging", k=4, metadata_boost={
            "domain_boost": 0.5,
            "domain_target": "radiology",
        })
        # d4 (radiology) should be ranked higher with boost
        boosted_scores = {e.id: s for e, s in results_yes[2]}
        unboosted_scores = {e.id: s for e, s in results_no[2]}
        # The boosted score for d4 should be higher than unboosted
        if "d4" in boosted_scores and "d4" in unboosted_scores:
            assert boosted_scores["d4"] > unboosted_scores["d4"]

    def test_mesh_boost_additive(self):
        rag = _build_rag_with_metadata()
        # d1 has 2 matching MeSH terms, d2 has 1
        results, _ = rag.retrieve_reranked("AI diagnosis", k=4, metadata_boost={
            "mesh_boost": 0.1,
            "mesh_target": ["Artificial Intelligence", "Diagnosis"],
        })
        scores = {e.id: s for e, s in results[2]}
        # d1 matches 2 terms, d2 matches 1 — d1 should get higher boost
        if "d1" in scores and "d2" in scores:
            # Get unboosted scores to compare the delta
            results_no, _ = rag.retrieve_reranked("AI diagnosis", k=4)
            scores_no = {e.id: s for e, s in results_no[2]}
            if "d1" in scores_no and "d2" in scores_no:
                d1_delta = scores["d1"] - scores_no["d1"]
                d2_delta = scores["d2"] - scores_no["d2"]
                assert d1_delta > d2_delta  # d1 gets more boost (2 matches vs 1)

    def test_recency_boost(self):
        rag = _build_rag_with_metadata()
        # d3 (2024) should get higher recency boost than d4 (2019)
        results, _ = rag.retrieve_reranked("medical research", k=4, metadata_boost={
            "recency_boost": 0.3,
        })
        scores = {e.id: s for e, s in results[2]}
        results_no, _ = rag.retrieve_reranked("medical research", k=4)
        scores_no = {e.id: s for e, s in results_no[2]}
        if "d3" in scores and "d4" in scores and "d3" in scores_no and "d4" in scores_no:
            d3_delta = scores["d3"] - scores_no["d3"]
            d4_delta = scores["d4"] - scores_no["d4"]
            assert d3_delta > d4_delta  # 2024 > 2019

    def test_no_boost_without_config(self):
        """Without metadata_boost, scores should be identical."""
        rag = _build_rag_with_metadata()
        results_a, _ = rag.retrieve_reranked("AI diagnosis", k=4)
        results_b, _ = rag.retrieve_reranked("AI diagnosis", k=4, metadata_boost=None)
        scores_a = {e.id: s for e, s in results_a[2]}
        scores_b = {e.id: s for e, s in results_b[2]}
        for doc_id in scores_a:
            assert abs(scores_a[doc_id] - scores_b.get(doc_id, 0.0)) < 1e-9

    def test_combined_filter_and_boost(self):
        """Filter narrows candidates, boost reranks within them."""
        rag = _build_rag_with_metadata()
        results, _ = rag.retrieve_reranked(
            "AI diagnosis", k=4,
            metadata_filters={"domain": "ai_diagnosis"},
            metadata_boost={"recency_boost": 0.5},
        )
        doc_ids = {e.id for e, _ in results[2]}
        # Only ai_diagnosis docs
        assert doc_ids.issubset({"d1", "d2"})
        # d1 (2023) should rank higher than d2 (2020) with recency boost
        scores = {e.id: s for e, s in results[2]}
        if "d1" in scores and "d2" in scores:
            assert scores["d1"] > scores["d2"]


# ============================================================
# Equivalence / regression tests
# ============================================================

class TestMetadataEquivalence:
    def test_no_filters_matches_baseline(self):
        """Calling retrieve with no filters should match calling without the param."""
        rag = _build_rag_with_metadata()
        r1, t1 = rag.retrieve_flat("AI diagnosis", k=4)
        r2, t2 = rag.retrieve_flat("AI diagnosis", k=4, metadata_filters=None)
        scores_1 = {e.id: s for e, s in r1[2]}
        scores_2 = {e.id: s for e, s in r2[2]}
        assert scores_1 == scores_2

    def test_save_load_preserves_filtered_results(self, tmp_path):
        """After save/load, filtered retrieval produces same results."""
        rag = _build_rag_with_metadata()
        db_path = tmp_path / "meta.db"
        save(db_path, rag)
        rag2 = load(db_path, HashEmbedding(dim=64))

        r1, _ = rag.retrieve_flat("AI", k=10, metadata_filters={"domain": "pharmacovigilance"})
        r2, _ = rag2.retrieve_flat("AI", k=10, metadata_filters={"domain": "pharmacovigilance"})

        ids1 = {e.id for e, _ in r1[2]}
        ids2 = {e.id for e, _ in r2[2]}
        assert ids1 == ids2
