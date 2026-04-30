"""Tests for CorpusProfiler and DomainGapScorer using real all-MiniLM-L6-v2."""

from __future__ import annotations

import pytest

from cam_rag.finetuning.backend import LocalSentenceTransformerBackend
from cam_rag.finetuning.profiler import (
    CorpusProfile,
    CorpusProfiler,
    DEFAULT_WEIGHTS,
    DomainGapReport,
    DomainGapScorer,
)
from cam_rag.rag.models import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def backend() -> LocalSentenceTransformerBackend:
    return LocalSentenceTransformerBackend(MODEL_NAME)


@pytest.fixture(scope="module")
def profiler(backend: LocalSentenceTransformerBackend) -> CorpusProfiler:
    return CorpusProfiler(backend, sample_size=0)


# ---------------------------------------------------------------------------
# Medical corpus (domain-specific, expected higher gap)
# ---------------------------------------------------------------------------

MEDICAL_CHUNKS = [
    Chunk(
        id="m1",
        document_id="doc_med",
        text=(
            "Thrombocytopenia is characterized by a platelet count below 150,000/μL. "
            "Immune thrombocytopenic purpura (ITP) involves autoantibody-mediated "
            "platelet destruction. Treatment includes corticosteroids, IVIG, and "
            "thrombopoietin receptor agonists such as eltrombopag and romiplostim."
        ),
        section_heading="Hematology",
    ),
    Chunk(
        id="m2",
        document_id="doc_med",
        text=(
            "Hepatorenal syndrome (HRS) develops in patients with advanced cirrhosis "
            "and portal hypertension. The pathophysiology involves splanchnic vasodilation, "
            "reduced effective arterial blood volume, and renal vasoconstriction. "
            "Terlipressin combined with albumin is the first-line pharmacotherapy."
        ),
        section_heading="Hepatology",
    ),
    Chunk(
        id="m3",
        document_id="doc_med",
        text=(
            "Pharmacokinetics of metformin: bioavailability 50-60%, Vd 654L, "
            "renal clearance 3.5x GFR. Contraindicated in eGFR<30 mL/min/1.73m2. "
            "HbA1c reduction 1.0-1.5% as monotherapy. Monitor for lactic acidosis."
        ),
        section_heading="Clinical Pharmacology",
    ),
    Chunk(
        id="m4",
        document_id="doc_med2",
        text=(
            "Acute ST-elevation myocardial infarction (STEMI) requires emergent "
            "percutaneous coronary intervention (PCI) within 90 minutes of first "
            "medical contact. Dual antiplatelet therapy with aspirin and P2Y12 "
            "inhibitor (ticagrelor or prasugrel) is standard of care."
        ),
        section_heading="Cardiology",
    ),
    Chunk(
        id="m5",
        document_id="doc_med2",
        text=(
            "Vancomycin trough levels should be maintained at 15-20 mg/L for "
            "serious MRSA infections. AUC/MIC ratio >400 is the preferred "
            "pharmacodynamic target. Nephrotoxicity risk increases with "
            "concomitant aminoglycosides or NSAIDs."
        ),
        section_heading="Infectious Disease",
    ),
]

# ---------------------------------------------------------------------------
# General corpus (common English, expected lower gap)
# ---------------------------------------------------------------------------

GENERAL_CHUNKS = [
    Chunk(
        id="g1",
        document_id="doc_gen",
        text=(
            "The weather forecast for this week shows sunny skies and warm "
            "temperatures. Highs will be in the upper 70s with light winds "
            "from the southwest. No rain is expected until next weekend."
        ),
    ),
    Chunk(
        id="g2",
        document_id="doc_gen",
        text=(
            "The local farmers market opens every Saturday morning. Fresh "
            "vegetables, fruits, and baked goods are available from dozens "
            "of vendors. Live music and food trucks add to the atmosphere."
        ),
    ),
    Chunk(
        id="g3",
        document_id="doc_gen",
        text=(
            "Regular exercise has many health benefits. Walking for thirty "
            "minutes a day can improve heart health and reduce stress. "
            "Swimming and cycling are also excellent choices for fitness."
        ),
    ),
    Chunk(
        id="g4",
        document_id="doc_gen2",
        text=(
            "The new library downtown has a wonderful children's section. "
            "Story time is held every Tuesday and Thursday afternoon. "
            "The building also features free internet access and study rooms."
        ),
    ),
    Chunk(
        id="g5",
        document_id="doc_gen2",
        text=(
            "Cooking at home saves money and is often healthier than eating "
            "out. Simple recipes with fresh ingredients can be prepared in "
            "under thirty minutes. Planning meals for the week helps too."
        ),
    ),
]


# ---------------------------------------------------------------------------
# CorpusProfiler tests
# ---------------------------------------------------------------------------


def test_profile_medical_corpus(profiler: CorpusProfiler) -> None:
    profile = profiler.profile(MEDICAL_CHUNKS)
    assert isinstance(profile, CorpusProfile)
    assert profile.num_chunks_sampled == len(MEDICAL_CHUNKS)
    assert profile.total_tokens_sampled > 0


def test_profile_general_corpus(profiler: CorpusProfiler) -> None:
    profile = profiler.profile(GENERAL_CHUNKS)
    assert isinstance(profile, CorpusProfile)
    assert profile.num_chunks_sampled == len(GENERAL_CHUNKS)


def test_medical_higher_oov_than_general(profiler: CorpusProfiler) -> None:
    med = profiler.profile(MEDICAL_CHUNKS)
    gen = profiler.profile(GENERAL_CHUNKS)
    assert med.oov_ratio > gen.oov_ratio, (
        f"Medical OOV {med.oov_ratio} should exceed general {gen.oov_ratio}"
    )


def test_medical_higher_fragmentation_than_general(profiler: CorpusProfiler) -> None:
    med = profiler.profile(MEDICAL_CHUNKS)
    gen = profiler.profile(GENERAL_CHUNKS)
    assert med.avg_subword_fragmentation > gen.avg_subword_fragmentation


def test_medical_higher_entity_density(profiler: CorpusProfiler) -> None:
    med = profiler.profile(MEDICAL_CHUNKS)
    gen = profiler.profile(GENERAL_CHUNKS)
    assert med.entity_density > gen.entity_density


def test_profile_empty_corpus(profiler: CorpusProfiler) -> None:
    profile = profiler.profile([])
    assert profile.num_chunks_sampled == 0
    assert profile.oov_ratio == 0.0
    assert profile.avg_subword_fragmentation == 0.0


def test_profile_single_chunk(profiler: CorpusProfiler) -> None:
    profile = profiler.profile([MEDICAL_CHUNKS[0]])
    assert profile.num_chunks_sampled == 1
    assert profile.embedding_dispersion == 0.0  # need >=2 chunks for pairwise


def test_profiler_sampling(backend: LocalSentenceTransformerBackend) -> None:
    """When sample_size < len(chunks), profiler samples reproducibly."""
    p = CorpusProfiler(backend, sample_size=3, seed=42)
    profile1 = p.profile(MEDICAL_CHUNKS)
    profile2 = p.profile(MEDICAL_CHUNKS)
    assert profile1.num_chunks_sampled == 3
    assert profile1.oov_ratio == profile2.oov_ratio  # deterministic


# ---------------------------------------------------------------------------
# DomainGapScorer tests
# ---------------------------------------------------------------------------


def test_gap_scorer_medical_vs_general(profiler: CorpusProfiler) -> None:
    scorer = DomainGapScorer()
    med_profile = profiler.profile(MEDICAL_CHUNKS)
    gen_profile = profiler.profile(GENERAL_CHUNKS)
    med_report = scorer.score(med_profile)
    gen_report = scorer.score(gen_profile)
    assert med_report.gap_score > gen_report.gap_score, (
        f"Medical gap {med_report.gap_score} should exceed general {gen_report.gap_score}"
    )


def test_gap_report_structure(profiler: CorpusProfiler) -> None:
    scorer = DomainGapScorer()
    report = scorer.score(profiler.profile(MEDICAL_CHUNKS))
    assert isinstance(report, DomainGapReport)
    assert 0.0 <= report.gap_score <= 1.0
    assert isinstance(report.recommendation, str)
    assert len(report.recommendation) > 0
    assert isinstance(report.profile, CorpusProfile)
    assert isinstance(report.signal_weights, dict)


def test_gap_recommendation_thresholds() -> None:
    """Verify the three recommendation levels work correctly."""
    scorer = DomainGapScorer()
    # Create synthetic profiles to trigger each threshold
    low_gap = CorpusProfile(
        oov_ratio=0.02,
        avg_subword_fragmentation=1.3,
        entity_density=0.01,
        embedding_dispersion=0.1,
        vocabulary_overlap=0.02,
        num_chunks_sampled=10,
        total_tokens_sampled=100,
    )
    high_gap = CorpusProfile(
        oov_ratio=0.4,
        avg_subword_fragmentation=3.5,
        entity_density=0.08,
        embedding_dispersion=0.9,
        vocabulary_overlap=0.4,
        num_chunks_sampled=10,
        total_tokens_sampled=100,
    )
    low_report = scorer.score(low_gap)
    high_report = scorer.score(high_gap)
    assert "adequate" in low_report.recommendation.lower()
    assert "strongly" in high_report.recommendation.lower()


def test_custom_weights() -> None:
    scorer = DomainGapScorer(weights={"oov_ratio": 1.0, "entity_density": 0.0})
    profile = CorpusProfile(
        oov_ratio=0.3,
        avg_subword_fragmentation=1.5,
        entity_density=0.5,
        embedding_dispersion=0.0,
        vocabulary_overlap=0.0,
        num_chunks_sampled=5,
        total_tokens_sampled=50,
    )
    report = scorer.score(profile)
    assert 0.0 <= report.gap_score <= 1.0
