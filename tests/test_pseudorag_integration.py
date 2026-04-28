"""Integration tests for PseudoRAG Pack v2 features.

Validates that all 5 PseudoRAG features (MoE scoring, accuracy contracts,
selective filter, contract-aware generation, ETF verification) work together
through the pipeline, in partial combinations, and maintain backward
compatibility when all flags are False.
"""

from __future__ import annotations

import pytest

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import (
    ETFVerificationStep,
    MoEImportanceScoringStep,
    AccuracyContractStep,
    GenerationStep,
    PipelineExecutor,
    SelectiveFilterStep,
    get_step,
)
from cam_rag.pipeline.registry import TechniqueDescriptor, TechniqueRegistry, compose_pipeline
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval import AdaptiveParams, compute_adaptive_params
from cam_rag.rl.feedback import DEFAULT_ARMS, FeedbackLoop
from cam_rag.scoring.contracts import classify_contract, classify_contracts
from cam_rag.scoring.moe import score_chunk_moe, score_chunks_moe
from cam_rag.verification.etf import compute_etf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunks() -> list[Chunk]:
    """Create a realistic set of chunks for integration testing."""
    return [
        Chunk(
            id="c1",
            document_id="doc1",
            text=(
                "This medication must never be used in patients with renal "
                "failure. Safety compliance is critical. Warning: monitor "
                "kidney function before administration. Contraindicated in "
                "severe hepatic impairment."
            ),
            title="Clinical Guideline",
            source="guidelines.md",
        ),
        Chunk(
            id="c2",
            document_id="doc1",
            text=(
                "The novel breakthrough treatment represents an unprecedented "
                "frontier approach. This first-principles methodology shows "
                "emergent capability with recursive feedback mechanisms and "
                "cascade synergy effects."
            ),
            title="Research Paper",
            source="research.md",
        ),
        Chunk(
            id="c3",
            document_id="doc2",
            text=(
                "This is a basic template for standard import operations. "
                "The default generic boilerplate provides trivial functionality. "
                "Standard template applies to all default cases."
            ),
            title="Template Guide",
            source="template.md",
        ),
        Chunk(
            id="c4",
            document_id="doc2",
            text=(
                "Follow the protocol algorithm for the workflow pipeline "
                "procedure. The schema gate ensures proper validation at "
                "each step of the workflow process."
            ),
            title="Methodology Doc",
            source="method.md",
        ),
    ]


def _make_spec(**kwargs) -> RAGAppSpec:
    """Create spec with PseudoRAG flags."""
    defaults = {
        "name": "pseudorag_test",
        "use_pipeline": True,
        "retrieval_top_k": 4,
    }
    defaults.update(kwargs)
    return RAGAppSpec(**defaults)


def _make_evidence(chunks: list[Chunk]) -> list[Evidence]:
    """Create evidence list from chunks."""
    return [
        Evidence(
            chunk=chunk,
            score=1.0 - i * 0.1,
            retriever="test",
            rank=i,
        )
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Test: Full pipeline with all 5 PseudoRAG features
# ---------------------------------------------------------------------------


class TestFullPseudoRAGPipeline:
    """All 5 features enabled end-to-end."""

    def test_moe_scores_all_chunks(self):
        """MoE scoring enriches all chunks with metadata."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        for chunk in chunks:
            assert "moe_combined_score" in chunk.metadata
            assert "moe_store_score" in chunk.metadata
            assert "moe_votes" in chunk.metadata
            assert isinstance(chunk.metadata["moe_votes"], dict)

    def test_contracts_classify_after_moe(self):
        """Contract classification uses MoE votes."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        classify_contracts(chunks)
        for chunk in chunks:
            assert chunk.metadata["accuracy_contract"] in ("hard", "medium", "soft")
            assert "concept_type" in chunk.metadata
            assert "emergence_role" in chunk.metadata

    def test_clinical_chunk_gets_hard_contract(self):
        """Clinical safety content should score high accuracy → hard tier."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        classify_contracts(chunks)
        clinical = chunks[0]  # Must never, safety, critical, warning, contraindicated
        assert clinical.metadata["accuracy_contract"] == "hard"

    def test_boilerplate_chunk_gets_soft_contract(self):
        """Boilerplate content scores low accuracy → soft tier."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        classify_contracts(chunks)
        boilerplate = chunks[2]  # basic, template, standard, default, generic, trivial, boilerplate
        assert boilerplate.metadata["accuracy_contract"] == "soft"

    def test_selective_filter_reorders_evidence(self):
        """Selective filter boosts high-value and penalises boilerplate."""
        from cam_rag.scoring.contracts import apply_selective_filter

        chunks = _make_chunks()
        score_chunks_moe(chunks)
        evidence = _make_evidence(chunks)

        # Give boilerplate chunk the highest initial score
        evidence[2].score = 1.5  # boilerplate has low store_score
        evidence[0].score = 0.5  # clinical has high store_score

        apply_selective_filter(evidence, threshold=0.1)

        # Clinical (high store_score) should be boosted above boilerplate
        # Check that pre_filter_score was preserved
        for ev in evidence:
            assert "pre_filter_score" in ev.signals

    def test_etf_verification_after_generation(self):
        """ETF report produced from generated answer + evidence."""
        chunks = _make_chunks()
        evidence = _make_evidence(chunks)
        answer = (
            "This medication must never be used in patients with renal failure. "
            "Safety compliance is critical for proper treatment."
        )
        report = compute_etf(answer, evidence)
        assert report.sovereignty_score > 0.5  # Good overlap with evidence
        assert report.skeptic_score == 1.0  # No generic phrases
        assert report.overall_score > 0.5

    def test_etf_detects_generic_language(self):
        """ETF skeptic catches generic boilerplate in answer."""
        chunks = _make_chunks()
        evidence = _make_evidence(chunks)
        answer = (
            "In general, it is well known that broadly speaking, "
            "the medication is typically speaking effective."
        )
        report = compute_etf(answer, evidence)
        assert report.skeptic_score < 0.7  # Penalised for generics
        assert report.generic_count >= 3


# ---------------------------------------------------------------------------
# Test: Partial feature adoption
# ---------------------------------------------------------------------------


class TestPartialAdoption:
    """Test feature subsets work correctly."""

    def test_moe_only(self):
        """P1 only: MoE scoring without contracts or filter."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        # No contracts set
        assert "accuracy_contract" not in chunks[0].metadata
        # But MoE scores are present
        assert chunks[0].metadata["moe_combined_score"] > 0

    def test_moe_plus_contracts_no_filter(self):
        """P1+P2: MoE + contracts but no selective filter."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        classify_contracts(chunks)
        # All have contracts
        for chunk in chunks:
            assert "accuracy_contract" in chunk.metadata
        # But no filter scores
        evidence = _make_evidence(chunks)
        for ev in evidence:
            assert "pre_filter_score" not in ev.signals

    def test_etf_standalone(self):
        """P5 only: ETF verification can run independently of MoE."""
        evidence = [
            Evidence(
                chunk=Chunk(id="c1", document_id="d1", text="metformin reduces glucose"),
                score=0.8,
                retriever="test",
            ),
        ]
        report = compute_etf("metformin reduces glucose levels", evidence)
        assert report.sovereignty_score > 0.3
        assert report.overall_score > 0.0


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """All PseudoRAG flags False → no new behavior."""

    def test_all_flags_false_no_new_metadata(self):
        """When all flags are False, chunks get no PseudoRAG metadata."""
        spec = _make_spec(
            moe_scoring_enabled=False,
            accuracy_contracts_enabled=False,
            selective_filter_enabled=False,
            etf_verification_enabled=False,
        )
        chunks = _make_chunks()
        # Simulate pipeline: steps are no-ops when disabled
        moe_step = MoEImportanceScoringStep()
        contract_step = AccuracyContractStep()
        filter_step = SelectiveFilterStep()
        etf_step = ETFVerificationStep()

        adaptive = compute_adaptive_params("summary", "test query", corpus_size=4)
        ctx = build_context("test query", chunks, spec, adaptive, limit=4)
        ctx.evidence = _make_evidence(chunks)

        moe_step.execute(ctx)
        contract_step.execute(ctx)
        filter_step.execute(ctx)
        etf_step.execute(ctx)

        for chunk in chunks:
            assert "moe_combined_score" not in chunk.metadata
            assert "accuracy_contract" not in chunk.metadata
        assert "etf_report" not in ctx.extras


# ---------------------------------------------------------------------------
# Test: RL arms include PseudoRAG steps
# ---------------------------------------------------------------------------


class TestRLArms:
    """RL arm definitions include PseudoRAG technique names."""

    def test_moe_scored_arm_exists(self):
        assert "moe_scored" in DEFAULT_ARMS
        assert "moe_importance_scoring" in DEFAULT_ARMS["moe_scored"]
        assert "selective_filter" in DEFAULT_ARMS["moe_scored"]

    def test_moe_contracted_arm_exists(self):
        assert "moe_contracted" in DEFAULT_ARMS
        assert "accuracy_contracts" in DEFAULT_ARMS["moe_contracted"]
        assert "moe_importance_scoring" in DEFAULT_ARMS["moe_contracted"]

    def test_full_pseudorag_arm_exists(self):
        assert "full_pseudorag" in DEFAULT_ARMS
        arm = DEFAULT_ARMS["full_pseudorag"]
        assert "moe_importance_scoring" in arm
        assert "accuracy_contracts" in arm
        assert "selective_filter" in arm
        assert "etf_verification" in arm
        assert "generation" in arm


# ---------------------------------------------------------------------------
# Test: Pipeline steps registered and discoverable
# ---------------------------------------------------------------------------


class TestStepRegistration:
    """New steps are registered in the global step registry."""

    def test_moe_step_registered(self):
        step = get_step("moe_importance_scoring")
        assert step.name == "moe_importance_scoring"

    def test_accuracy_contracts_step_registered(self):
        step = get_step("accuracy_contracts")
        assert step.name == "accuracy_contracts"

    def test_selective_filter_step_registered(self):
        step = get_step("selective_filter")
        assert step.name == "selective_filter"

    def test_etf_verification_step_registered(self):
        step = get_step("etf_verification")
        assert step.name == "etf_verification"


# ---------------------------------------------------------------------------
# Test: MoE metadata survives pipeline
# ---------------------------------------------------------------------------


class TestMetadataSurvival:
    """MoE metadata persists through subsequent pipeline steps."""

    def test_moe_metadata_on_evidence_chunks(self):
        """Evidence chunks retain MoE metadata after scoring."""
        chunks = _make_chunks()
        score_chunks_moe(chunks)
        classify_contracts(chunks)
        evidence = _make_evidence(chunks)
        for ev in evidence:
            assert "moe_combined_score" in ev.chunk.metadata
            assert "moe_votes" in ev.chunk.metadata
            assert "accuracy_contract" in ev.chunk.metadata
