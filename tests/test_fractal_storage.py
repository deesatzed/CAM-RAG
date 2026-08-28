"""
Tests for fractrag.storage — SQLite persistence layer.

All tests use real HashEmbedding computations. No mocks.
"""

import os
import numpy as np
import pytest

from cam_rag.retrieval.fractal import (
    FractalRAG,
    HashEmbedding,
    DocumentProfile,
    IndexEntry,
)
from cam_rag.retrieval.fractal.storage import save, load, DimensionMismatchError


# ============================================================
# Helpers
# ============================================================

def _make_engine(dim=64, docs=None, profiles=None, adapter_strength=0.25):
    """Build a FractalRAG with real HashEmbedding and optional docs/profiles."""
    rag = FractalRAG(backend=HashEmbedding(dim=dim), adapter_strength=adapter_strength)
    docs = docs or {}
    profiles = profiles or {}
    for doc_id, text in docs.items():
        prof = profiles.get(doc_id)
        rag.add_document(doc_id, text, profile=prof)
    return rag


SAMPLE_DOCS = {
    "ai_overview": (
        "Artificial intelligence is transforming every industry. "
        "Machine learning allows systems to learn patterns from data without explicit programming. "
        "Neural networks, inspired by the human brain, power modern deep learning breakthroughs. "
        "Large language models like GPT and Claude represent the current frontier."
    ),
    "biology_dna": (
        "DNA serves as the fundamental blueprint of all known life forms. "
        "The double helix structure, discovered by Watson and Crick, encodes genetic information. "
        "During cell division, mitosis ensures accurate replication of chromosomes. "
        "Evolution acts through natural selection on genetic variation over generations."
    ),
    "history_rome": (
        "The Western Roman Empire collapsed in 476 AD after centuries of decline. "
        "Economic troubles, barbarian invasions, and political instability were key factors. "
        "The Renaissance later revived classical art, science, and humanism in Europe. "
        "World War II, ending in 1945, reshaped global power structures forever."
    ),
}

SAMPLE_PROFILES = {
    "ai_overview": DocumentProfile(
        accuracy_importance="high", complexity_level="medium", domain="technology"
    ),
    "biology_dna": DocumentProfile(
        accuracy_importance="high", complexity_level="high", domain="science"
    ),
}



SAMPLE_DOCS_FIXTURE = SAMPLE_DOCS


@pytest.fixture
def unified_engine_loaded():
    """FractalRAG engine preloaded with the 3 sample docs."""
    rag = FractalRAG(backend=HashEmbedding(dim=64))
    for doc_id, text in SAMPLE_DOCS.items():
        rag.add_document(doc_id, text)
    return rag


# ============================================================
# Round-trip tests
# ============================================================

class TestRoundTrip:
    """Save then load, verify all state matches."""

    def test_basic_round_trip(self, tmp_path, unified_engine_loaded):
        """Save and load restores docs, index counts, adapters, derivatives."""
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag.stats() == rag2.stats()

    def test_documents_preserved(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag2.docs == rag.docs

    def test_index_vectors_match(self, tmp_path, unified_engine_loaded):
        """Every index vector round-trips with float32 precision."""
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        for level in [0, 1, 2]:
            assert len(rag2.index[level]) == len(rag.index[level])
            for orig, loaded in zip(rag.index[level], rag2.index[level]):
                assert orig.id == loaded.id
                assert orig.level == loaded.level
                assert orig.text == loaded.text
                assert orig.parent == loaded.parent
                np.testing.assert_array_almost_equal(orig.vec, loaded.vec, decimal=6)

    def test_adapters_match(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert set(rag2.adapters.keys()) == set(rag.adapters.keys())
        for doc_id in rag.adapters:
            np.testing.assert_array_almost_equal(
                rag.adapters[doc_id], rag2.adapters[doc_id], decimal=6
            )

    def test_derivatives_match(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert set(rag2.derivatives.keys()) == set(rag.derivatives.keys())
        for sid in rag.derivatives:
            for key in ["d1", "d2"]:
                np.testing.assert_array_almost_equal(
                    rag.derivatives[sid][key],
                    rag2.derivatives[sid][key],
                    decimal=6,
                )

    def test_adapter_strength_preserved(self, tmp_path):
        """Non-default adapter_strength round-trips correctly."""
        rag = _make_engine(dim=64, docs={"a": "Test doc."}, adapter_strength=0.42)
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag2._adapter_strength == pytest.approx(0.42)


# ============================================================
# Profile round-trip
# ============================================================

class TestProfileRoundTrip:
    def test_profiles_preserved(self, tmp_path):
        rag = _make_engine(dim=64, docs=SAMPLE_DOCS, profiles=SAMPLE_PROFILES)
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert set(rag2.profiles.keys()) == set(rag.profiles.keys())
        for doc_id in rag.profiles:
            orig = rag.profiles[doc_id]
            loaded = rag2.profiles[doc_id]
            assert orig.accuracy_importance == loaded.accuracy_importance
            assert orig.complexity_level == loaded.complexity_level
            assert orig.update_frequency == loaded.update_frequency
            assert orig.domain == loaded.domain
            assert orig.likely_question_types == loaded.likely_question_types
            assert orig.tolerance_for_hallucination == loaded.tolerance_for_hallucination
            assert orig.priority == loaded.priority
            assert orig.mission_critical == loaded.mission_critical

    def test_profile_question_types_list(self, tmp_path):
        """likely_question_types list with multiple entries round-trips."""
        prof = DocumentProfile(
            likely_question_types=["specification", "summary", "logic"]
        )
        rag = _make_engine(dim=64, docs={"doc1": "Test."}, profiles={"doc1": prof})
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag2.profiles["doc1"].likely_question_types == [
            "specification", "summary", "logic"
        ]


# ============================================================
# Retrieval equivalence
# ============================================================

class TestRetrievalEquivalence:
    """Results from a loaded engine must match the original exactly."""

    def test_retrieve_equivalence(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded
        query = "How does machine learning work?"

        results_before, qtype_before = rag.retrieve(query, k=3)

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))
        results_after, qtype_after = rag2.retrieve(query, k=3)

        assert qtype_before == qtype_after
        for level in results_before:
            assert len(results_before[level]) == len(results_after[level])
            for (e1, s1), (e2, s2) in zip(results_before[level], results_after[level]):
                assert e1.id == e2.id
                assert s1 == pytest.approx(s2, abs=1e-6)

    def test_retrieve_reranked_equivalence(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded
        query = "Summarize DNA and evolution."

        results_before, qtype_before = rag.retrieve_reranked(query, k=3)

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))
        results_after, qtype_after = rag2.retrieve_reranked(query, k=3)

        assert qtype_before == qtype_after
        for level in results_before:
            assert len(results_before[level]) == len(results_after[level])
            for (e1, s1), (e2, s2) in zip(results_before[level], results_after[level]):
                assert e1.id == e2.id
                assert s1 == pytest.approx(s2, abs=1e-6)

    def test_retrieve_adaptive_equivalence(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded
        query = "What are the overall themes across technology and history?"

        results_before, qtype_before = rag.retrieve_adaptive(query, k=3)

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))
        results_after, qtype_after = rag2.retrieve_adaptive(query, k=3)

        assert qtype_before == qtype_after
        for level in results_before:
            assert len(results_before[level]) == len(results_after[level])
            for (e1, s1), (e2, s2) in zip(results_before[level], results_after[level]):
                assert e1.id == e2.id
                assert s1 == pytest.approx(s2, abs=1e-6)

    def test_retrieve_flat_equivalence(self, tmp_path, unified_engine_loaded):
        db = tmp_path / "test.db"
        rag = unified_engine_loaded
        query = "What is DNA?"

        results_before, qtype_before = rag.retrieve_flat(query, k=3)

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))
        results_after, qtype_after = rag2.retrieve_flat(query, k=3)

        assert qtype_before == qtype_after
        for level in results_before:
            assert len(results_before[level]) == len(results_after[level])
            for (e1, s1), (e2, s2) in zip(results_before[level], results_after[level]):
                assert e1.id == e2.id
                assert s1 == pytest.approx(s2, abs=1e-6)


# ============================================================
# Incremental add
# ============================================================

class TestIncrementalAdd:
    def test_add_after_load_then_save(self, tmp_path, unified_engine_loaded):
        """Load, add a new doc, save again, load again — all docs present."""
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        new_text = (
            "Quantum computing uses qubits that exist in superposition. "
            "This allows exponential speedup for certain problem classes."
        )
        rag2.add_document("quantum", new_text)
        save(db, rag2)

        rag3 = load(db, backend=HashEmbedding(dim=64))
        assert "quantum" in rag3.docs
        assert rag3.docs["quantum"] == new_text
        assert rag3.stats()["documents"] == 4
        # Original docs still present
        for doc_id in SAMPLE_DOCS:
            assert doc_id in rag3.docs


# ============================================================
# Dimension mismatch
# ============================================================

class TestDimensionMismatch:
    def test_dim_mismatch_raises(self, tmp_path):
        rag = _make_engine(dim=64, docs={"a": "Test."})
        db = tmp_path / "test.db"

        save(db, rag)

        with pytest.raises(DimensionMismatchError, match="dim=128.*stored dim=64"):
            load(db, backend=HashEmbedding(dim=128))

    def test_same_dim_different_backend_ok(self, tmp_path):
        """Same dim but conceptually different backend should load fine."""
        rag = _make_engine(dim=64, docs={"a": "Test."})
        db = tmp_path / "test.db"

        save(db, rag)

        # Load with a fresh HashEmbedding of same dim — should work
        rag2 = load(db, backend=HashEmbedding(dim=64))
        assert rag2.stats()["documents"] == 1


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_engine(self, tmp_path):
        """Empty engine (no docs) round-trips successfully."""
        rag = _make_engine(dim=64)
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag2.stats()["documents"] == 0
        assert rag2.stats() == rag.stats()

    def test_unicode_text(self, tmp_path):
        """Unicode text in documents and index entries round-trips."""
        text = "Les réseaux neuronaux sont puissants. 日本語テスト。 Ñoño 🧬 émojis."
        rag = _make_engine(dim=64, docs={"unicode_doc": text})
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert rag2.docs["unicode_doc"] == text

    def test_overwrite_existing_file(self, tmp_path, unified_engine_loaded):
        """Saving to an existing db file overwrites it completely."""
        db = tmp_path / "test.db"
        rag = unified_engine_loaded

        save(db, rag)
        stats1 = load(db, backend=HashEmbedding(dim=64)).stats()

        # Save a smaller engine to the same path
        rag_small = _make_engine(dim=64, docs={"only_one": "Single doc."})
        save(db, rag_small)
        rag_reloaded = load(db, backend=HashEmbedding(dim=64))

        assert rag_reloaded.stats()["documents"] == 1
        assert "only_one" in rag_reloaded.docs
        assert "ai_overview" not in rag_reloaded.docs

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load(tmp_path / "nonexistent.db", backend=HashEmbedding(dim=64))

    def test_docs_without_profiles(self, tmp_path):
        """Documents added without profiles have empty profiles dict."""
        rag = _make_engine(dim=64, docs=SAMPLE_DOCS)
        db = tmp_path / "test.db"

        save(db, rag)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        assert len(rag2.profiles) == 0
        assert rag2.stats()["documents"] == 3

    def test_vectors_are_writable(self, tmp_path, unified_engine_loaded):
        """Loaded vectors must be writable (not read-only from np.frombuffer)."""
        db = tmp_path / "test.db"
        save(db, unified_engine_loaded)
        rag2 = load(db, backend=HashEmbedding(dim=64))

        # Should not raise — vectors must be mutable
        for entry in rag2.index[0]:
            entry.vec[0] = 0.0  # Would raise if read-only

        for sid, derivs in rag2.derivatives.items():
            derivs["d1"][0] = 0.0
            derivs["d2"][0] = 0.0

        for doc_id, vec in rag2.adapters.items():
            vec[0] = 0.0
