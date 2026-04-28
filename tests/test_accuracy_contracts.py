"""Tests for accuracy contracts and selective filtering."""

import pytest

from cam_rag.rag.models import Chunk, Evidence
from cam_rag.scoring.contracts import (
    ContractTier,
    ConceptType,
    EmergenceRole,
    classify_contract,
    classify_contracts,
    apply_selective_filter,
)


def _make_chunk_with_moe(
    text: str = "Reciprocal rank fusion combines sparse and dense retrieval signals",
    accuracy: float = 0.0,
    novelty: float = 0.0,
    methodology: float = 0.0,
    context_sovereignty: float = 0.0,
    reproducibility: float = 0.0,
    emergence_leverage: float = 0.0,
    usability: float = 0.0,
    store_score: float = 0.0,
    chunk_id: str = "c1",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc1",
        text=text,
        metadata={
            "moe_votes": {
                "accuracy_criticality": accuracy,
                "novelty": novelty,
                "methodology": methodology,
                "context_sovereignty": context_sovereignty,
                "reproducibility": reproducibility,
                "emergence_leverage": emergence_leverage,
                "usability": usability,
            },
            "moe_store_score": store_score,
            "moe_combined_score": 0.5,
        },
    )


def _make_evidence(
    chunk: Chunk,
    score: float = 0.8,
    retriever: str = "rrf_fusion",
    rank: int = 0,
) -> Evidence:
    return Evidence(
        chunk=chunk,
        score=score,
        retriever=retriever,
        rank=rank,
    )


# --------------------------------------------------------------------------
# 1. Tier classification boundary tests
# --------------------------------------------------------------------------


class TestTierClassification:
    """Contract tier classification based on accuracy_criticality score."""

    def test_hard_tier_at_boundary(self):
        """accuracy_criticality=0.6 should classify as hard."""
        chunk = _make_chunk_with_moe(
            text="Cross-encoder reranking requires pairwise scoring of query-document pairs",
            accuracy=0.6,
        )
        tier = classify_contract(chunk)
        assert tier == "hard"

    def test_medium_tier_at_boundary(self):
        """accuracy_criticality=0.35 should classify as medium."""
        chunk = _make_chunk_with_moe(
            text="BM25 term frequency scoring uses Robertson-Sparck Jones weighting",
            accuracy=0.35,
        )
        tier = classify_contract(chunk)
        assert tier == "medium"

    def test_soft_tier_below_medium(self):
        """accuracy_criticality=0.3 should classify as soft."""
        chunk = _make_chunk_with_moe(
            text="The system returns relevant documents based on the user query",
            accuracy=0.3,
        )
        tier = classify_contract(chunk)
        assert tier == "soft"


# --------------------------------------------------------------------------
# 2. Concept type inference tests
# --------------------------------------------------------------------------


class TestConceptTypeInference:
    """Concept type inferred from MoE vote profile."""

    def test_context_sovereignty_rule(self):
        """context_sovereignty >= 0.45 should yield context_sovereignty_rule."""
        chunk = _make_chunk_with_moe(
            text="All retrieval results must be confined to the uploaded document corpus",
            context_sovereignty=0.45,
        )
        classify_contract(chunk)
        assert chunk.metadata["concept_type"] == "context_sovereignty_rule"

    def test_methodology_type(self):
        """methodology >= 0.5 should yield methodology."""
        chunk = _make_chunk_with_moe(
            text="Apply reciprocal rank fusion with k=60 across all retrieval sources",
            methodology=0.5,
        )
        classify_contract(chunk)
        assert chunk.metadata["concept_type"] == "methodology"

    def test_frontier_concept_via_novelty(self):
        """novelty >= 0.5 should yield frontier_concept."""
        chunk = _make_chunk_with_moe(
            text="Emergent retrieval patterns arise from multi-hop graph traversal",
            novelty=0.5,
        )
        classify_contract(chunk)
        assert chunk.metadata["concept_type"] == "frontier_concept"

    def test_operational_hint_when_all_low(self):
        """All scores below thresholds should yield operational_hint."""
        chunk = _make_chunk_with_moe(
            text="The configuration file is located in the project root directory",
            novelty=0.1,
            methodology=0.2,
            context_sovereignty=0.1,
            emergence_leverage=0.1,
        )
        classify_contract(chunk)
        assert chunk.metadata["concept_type"] == "operational_hint"


# --------------------------------------------------------------------------
# 3. Emergence role inference tests
# --------------------------------------------------------------------------


class TestEmergenceRoleInference:
    """Emergence role inferred from MoE vote profile."""

    def test_constrain_role(self):
        """context_sovereignty >= 0.45 should yield constrain."""
        chunk = _make_chunk_with_moe(
            text="Never retrieve from external knowledge bases outside the document scope",
            context_sovereignty=0.45,
        )
        classify_contract(chunk)
        assert chunk.metadata["emergence_role"] == "constrain"

    def test_route_role(self):
        """methodology >= 0.5 should yield route."""
        chunk = _make_chunk_with_moe(
            text="Use dense retrieval for semantic similarity and sparse for exact match",
            methodology=0.5,
        )
        classify_contract(chunk)
        assert chunk.metadata["emergence_role"] == "route"

    def test_seed_role_via_emergence_leverage(self):
        """emergence_leverage >= 0.5 should yield seed."""
        chunk = _make_chunk_with_moe(
            text="Graph-based retrieval enables discovery of latent document relationships",
            emergence_leverage=0.5,
        )
        classify_contract(chunk)
        assert chunk.metadata["emergence_role"] == "seed"

    def test_support_role_when_all_low(self):
        """All scores below thresholds should yield support."""
        chunk = _make_chunk_with_moe(
            text="Chunk overlap improves boundary coverage for paragraph-level splits",
            novelty=0.1,
            methodology=0.2,
            context_sovereignty=0.1,
            emergence_leverage=0.1,
        )
        classify_contract(chunk)
        assert chunk.metadata["emergence_role"] == "support"


# --------------------------------------------------------------------------
# 4. Metadata side-effects tests
# --------------------------------------------------------------------------


class TestMetadataSideEffects:
    """classify_contract sets all three metadata keys on the chunk."""

    def test_all_keys_set_hard_tier(self):
        """Hard tier chunk should have all three metadata keys populated."""
        chunk = _make_chunk_with_moe(
            text="Citation grounding requires verbatim evidence from source documents",
            accuracy=0.75,
            context_sovereignty=0.5,
        )
        classify_contract(chunk)
        assert chunk.metadata["accuracy_contract"] == "hard"
        assert chunk.metadata["concept_type"] == "context_sovereignty_rule"
        assert chunk.metadata["emergence_role"] == "constrain"

    def test_all_keys_set_soft_tier(self):
        """Soft tier chunk should still have all three metadata keys."""
        chunk = _make_chunk_with_moe(
            text="General information about how the platform works",
            accuracy=0.1,
            methodology=0.6,
        )
        classify_contract(chunk)
        assert chunk.metadata["accuracy_contract"] == "soft"
        assert chunk.metadata["concept_type"] == "methodology"
        assert chunk.metadata["emergence_role"] == "route"


# --------------------------------------------------------------------------
# 5. Batch processing test
# --------------------------------------------------------------------------


class TestBatchProcessing:
    """classify_contracts works on a list of chunks."""

    def test_classify_contracts_batch(self):
        """classify_contracts processes all chunks and returns the same list."""
        chunks = [
            _make_chunk_with_moe(
                text="Semantic chunking preserves sentence boundaries",
                accuracy=0.7,
                chunk_id="c1",
            ),
            _make_chunk_with_moe(
                text="Paragraph splitting uses double newline delimiters",
                accuracy=0.4,
                chunk_id="c2",
            ),
            _make_chunk_with_moe(
                text="Default chunk size is 512 tokens",
                accuracy=0.2,
                chunk_id="c3",
            ),
        ]
        result = classify_contracts(chunks)
        assert result is chunks
        assert chunks[0].metadata["accuracy_contract"] == "hard"
        assert chunks[1].metadata["accuracy_contract"] == "medium"
        assert chunks[2].metadata["accuracy_contract"] == "soft"


# --------------------------------------------------------------------------
# 6. Score adjustment above threshold
# --------------------------------------------------------------------------


class TestScoreAdjustmentAboveThreshold:
    """Multiplier > 1.0 for chunks with store_score above threshold."""

    def test_high_store_score_boosts_evidence(self):
        """store_score=0.8 with threshold=0.55 should boost by 1.25x."""
        chunk = _make_chunk_with_moe(
            text="Dense vector retrieval uses cosine similarity in embedding space",
            store_score=0.8,
            accuracy=0.3,
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = 1.0 + (0.8 - 0.55) * 1.0 = 1.25
        assert ev.score == pytest.approx(1.25, abs=1e-6)

    def test_very_high_store_score_capped_at_1_5(self):
        """store_score=1.0 should cap the multiplier at 1.5."""
        chunk = _make_chunk_with_moe(
            text="Multi-hop retrieval synthesizes evidence across document boundaries",
            store_score=1.0,
            accuracy=0.3,
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = 1.0 + (1.0 - 0.55) * 1.0 = 1.45 (under cap)
        assert ev.score == pytest.approx(1.45, abs=1e-6)


# --------------------------------------------------------------------------
# 7. Score adjustment below threshold
# --------------------------------------------------------------------------


class TestScoreAdjustmentBelowThreshold:
    """Multiplier < 1.0 for chunks with store_score below threshold."""

    def test_low_store_score_penalises_evidence(self):
        """store_score=0.2 with threshold=0.55 should penalise."""
        chunk = _make_chunk_with_moe(
            text="This section describes general setup instructions",
            store_score=0.2,
            accuracy=0.1,
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = 0.5 + (0.2 / 0.55) * 0.5 = 0.5 + 0.18181... = 0.68181...
        expected = 0.5 + (0.2 / 0.55) * 0.5
        assert ev.score == pytest.approx(expected, abs=1e-6)

    def test_zero_store_score_minimum_multiplier(self):
        """store_score=0.0 should yield multiplier of 0.5."""
        chunk = _make_chunk_with_moe(
            text="Boilerplate copyright notice for the document",
            store_score=0.0,
            accuracy=0.1,
        )
        ev = _make_evidence(chunk, score=2.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = 0.5 + (0.0 / 0.55) * 0.5 = 0.5
        assert ev.score == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# 8. Hard override test
# --------------------------------------------------------------------------


class TestHardOverride:
    """accuracy_criticality >= 0.82 AND reproducibility < 0.70 forces max multiplier."""

    def test_hard_override_applies_max_boost(self):
        """High accuracy + low reproducibility triggers 1.5x override."""
        chunk = _make_chunk_with_moe(
            text="Clinical dosage calculations require exact precision from source tables",
            accuracy=0.85,
            reproducibility=0.5,
            store_score=0.3,  # normally would penalise
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # Hard override: multiplier = 1.5 regardless of store_score
        assert ev.score == pytest.approx(1.5, abs=1e-6)


# --------------------------------------------------------------------------
# 9. Evidence re-sorting test
# --------------------------------------------------------------------------


class TestEvidenceReSorting:
    """After filter, evidence is sorted by adjusted score descending."""

    def test_reorder_after_filter(self):
        """Lower-ranked chunk with high store_score should move to top."""
        high_value_chunk = _make_chunk_with_moe(
            text="Confidence scoring aggregates lexical overlap and embedding similarity",
            store_score=0.9,
            accuracy=0.5,
            chunk_id="c_high",
        )
        low_value_chunk = _make_chunk_with_moe(
            text="Table of contents for the document appendix section",
            store_score=0.1,
            accuracy=0.1,
            chunk_id="c_low",
        )
        ev_low = _make_evidence(low_value_chunk, score=1.0, rank=0)
        ev_high = _make_evidence(high_value_chunk, score=0.9, rank=1)

        result = apply_selective_filter([ev_low, ev_high], threshold=0.55)

        # After filtering, high-value chunk should be ranked first
        assert result[0].chunk.id == "c_high"
        assert result[1].chunk.id == "c_low"
        assert result[0].rank == 0
        assert result[1].rank == 1


# --------------------------------------------------------------------------
# 10. No-op when no MoE metadata
# --------------------------------------------------------------------------


class TestNoMoEMetadata:
    """Missing moe_votes defaults to soft tier and 0.0 store_score in filter."""

    def test_missing_moe_votes_defaults_soft(self):
        """Chunk without moe_votes should be classified as soft."""
        chunk = Chunk(
            id="bare",
            document_id="doc1",
            text="A plain chunk without any scoring metadata attached",
            metadata={},
        )
        tier = classify_contract(chunk)
        assert tier == "soft"
        assert chunk.metadata["concept_type"] == "operational_hint"
        assert chunk.metadata["emergence_role"] == "support"


# --------------------------------------------------------------------------
# 11. Pre_filter_score signal preservation
# --------------------------------------------------------------------------


class TestPreFilterScoreSignal:
    """Original score is preserved in signals before adjustment."""

    def test_original_score_stored_in_signals(self):
        """signals['pre_filter_score'] should contain the original evidence score."""
        chunk = _make_chunk_with_moe(
            text="Query expansion adds synonyms to improve recall coverage",
            store_score=0.7,
            accuracy=0.4,
        )
        ev = _make_evidence(chunk, score=0.85)
        apply_selective_filter([ev], threshold=0.55)
        assert ev.signals["pre_filter_score"] == pytest.approx(0.85, abs=1e-6)
        # Score should be different from original after adjustment
        assert ev.score != pytest.approx(0.85, abs=1e-6)


# --------------------------------------------------------------------------
# 12. Multiplier range bounds
# --------------------------------------------------------------------------


class TestMultiplierRange:
    """Multiplier is always bounded within [0.5, 1.5]."""

    def test_multiplier_never_exceeds_1_5(self):
        """Even with extremely high store_score the multiplier caps at 1.5."""
        chunk = _make_chunk_with_moe(
            text="Critical retrieval pathway requires exact document provenance",
            store_score=2.0,  # artificially high
            accuracy=0.3,
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = min(1.0 + (2.0 - 0.55), 1.5) = min(2.45, 1.5) = 1.5
        assert ev.score == pytest.approx(1.5, abs=1e-6)

    def test_multiplier_never_below_0_5(self):
        """store_score=0.0 yields minimum multiplier of 0.5."""
        chunk = _make_chunk_with_moe(
            text="Empty boilerplate header with no substantive content",
            store_score=0.0,
            accuracy=0.1,
        )
        ev = _make_evidence(chunk, score=1.0)
        apply_selective_filter([ev], threshold=0.55)
        # multiplier = 0.5 + (0.0/0.55)*0.5 = 0.5
        assert ev.score == pytest.approx(0.5, abs=1e-6)
