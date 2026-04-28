"""Tests for MoE importance scoring.

All chunks use real text content -- no mock data or placeholders.
"""

from __future__ import annotations

import pytest
from cam_rag.rag.models import Chunk
from cam_rag.scoring.moe import (
    ExpertVote,
    MoEResult,
    _EXPERTS,
    _score_expert,
    _tokenize,
    score_chunk_moe,
    score_chunks_moe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, **metadata: object) -> Chunk:
    """Create a real Chunk with the given text and optional metadata."""
    return Chunk(
        id="c-test",
        document_id="d-test",
        text=text,
        metadata=dict(metadata),
    )


# ---------------------------------------------------------------------------
# Sample texts (real, not mock)
# ---------------------------------------------------------------------------

CLINICAL_TEXT = (
    "This medication must never be used in patients with known allergy. "
    "Safety compliance is critical for contraindicated drugs. "
    "Warning: check safety protocols."
)

BOILERPLATE_TEXT = (
    "This is a basic template for standard import operations. "
    "The default generic boilerplate is trivial."
)

NOVEL_TEXT = (
    "This novel breakthrough represents an unprecedented frontier "
    "approach using first-principles analysis."
)

METHODOLOGY_TEXT = (
    "Follow the protocol algorithm to execute the workflow "
    "pipeline procedure."
)

CONTEXT_TEXT = (
    "The evidence is grounded by citation reference. "
    "Context is verified with provenance tracking."
)

EMERGENCE_TEXT = (
    "The capability is emergent from feedback recursive leverage. "
    "Cascade synergy amplifies the result."
)

USABILITY_TEXT = (
    "Run the gate prompt command to score the test. "
    "Validator output is actionable and executable."
)


# ===================================================================
# 1. Individual expert tests (7 tests)
# ===================================================================

class TestIndividualExperts:
    """Each expert scores its own keywords correctly."""

    def test_accuracy_criticality_expert(self) -> None:
        tokens = _tokenize(CLINICAL_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "accuracy_criticality")
        score, hits = _score_expert(expert, tokens)
        # CLINICAL_TEXT contains: must, never, safety (x2), compliance, critical, contraindicated, warning
        assert hits >= 5
        assert score > 0.0

    def test_novelty_expert(self) -> None:
        tokens = _tokenize(NOVEL_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "novelty")
        score, hits = _score_expert(expert, tokens)
        # novel, breakthrough, unprecedented, frontier, first-principles
        assert hits >= 4
        assert score > 0.0

    def test_methodology_expert(self) -> None:
        tokens = _tokenize(METHODOLOGY_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "methodology")
        score, hits = _score_expert(expert, tokens)
        # protocol, algorithm, workflow, pipeline, procedure
        assert hits >= 5
        assert score >= 1.0

    def test_context_sovereignty_expert(self) -> None:
        tokens = _tokenize(CONTEXT_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "context_sovereignty")
        score, hits = _score_expert(expert, tokens)
        # evidence, grounded, citation, reference, context, verified, provenance
        assert hits >= 5
        assert score > 0.0

    def test_reproducibility_expert(self) -> None:
        tokens = _tokenize(BOILERPLATE_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "reproducibility")
        score, hits = _score_expert(expert, tokens)
        # basic, template, standard, import, default, generic, boilerplate, trivial
        assert hits >= 5
        assert score > 0.0

    def test_emergence_leverage_expert(self) -> None:
        tokens = _tokenize(EMERGENCE_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "emergence_leverage")
        score, hits = _score_expert(expert, tokens)
        # capability, emergent, feedback, recursive, leverage, cascade, synergy
        assert hits >= 5
        assert score > 0.0

    def test_usability_expert(self) -> None:
        tokens = _tokenize(USABILITY_TEXT)
        expert = next(e for e in _EXPERTS if e.name == "usability")
        score, hits = _score_expert(expert, tokens)
        # gate, prompt, command, score, test, validator, actionable, executable
        assert hits >= 5
        assert score > 0.0


# ===================================================================
# 2. Combined score calculation (2 tests)
# ===================================================================

class TestCombinedScore:
    """Weighted average is calculated correctly."""

    def test_combined_score_positive_for_keyword_rich_text(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        assert result.combined_score > 0.0

    def test_combined_score_is_weighted_average(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        total_w = sum(v.weight for v in result.votes.values())
        expected = sum(
            v.weight * v.score for v in result.votes.values()
        ) / total_w
        assert abs(result.combined_score - round(expected, 6)) < 1e-6


# ===================================================================
# 3. Store score formula (2 tests)
# ===================================================================

class TestStoreScore:
    """Linear store-score formula produces expected values."""

    def test_store_score_positive_for_clinical_text(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        # accuracy_criticality is high, reproducibility is low -> positive
        assert result.store_score > 0.0

    def test_store_score_matches_formula(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT, complexity_score=0.5)
        result = score_chunk_moe(chunk)
        v = result.votes
        expected = (
            0.25 * v["accuracy_criticality"].score
            + 0.20 * v["novelty"].score
            + 0.20 * v["methodology"].score
            + 0.15 * v["emergence_leverage"].score
            + 0.10 * v["context_sovereignty"].score
            - 0.30 * v["reproducibility"].score
            + 0.10 * 0.5
        )
        assert abs(result.store_score - round(expected, 6)) < 1e-6


# ===================================================================
# 4. Chunk metadata side-effects (3 tests)
# ===================================================================

class TestMetadataSideEffects:
    """score_chunk_moe writes expected metadata keys."""

    def test_moe_combined_score_set(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        assert chunk.metadata["moe_combined_score"] == result.combined_score

    def test_moe_store_score_set(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        assert chunk.metadata["moe_store_score"] == result.store_score

    def test_moe_votes_set(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        votes_meta = chunk.metadata["moe_votes"]
        assert isinstance(votes_meta, dict)
        for name, vote in result.votes.items():
            assert votes_meta[name] == vote.score


# ===================================================================
# 5. Empty / whitespace text (2 tests)
# ===================================================================

class TestEmptyText:
    """Degenerate inputs return zero scores."""

    def test_empty_string(self) -> None:
        chunk = _make_chunk("")
        result = score_chunk_moe(chunk)
        assert result.combined_score == 0.0
        assert result.store_score == 0.0

    def test_whitespace_only(self) -> None:
        chunk = _make_chunk("   \n\t  ")
        result = score_chunk_moe(chunk)
        assert result.combined_score == 0.0
        assert result.store_score == 0.0


# ===================================================================
# 6. Reproducibility penalty (1 test)
# ===================================================================

class TestReproducibilityPenalty:
    """Boilerplate text receives a lower (possibly negative) store score."""

    def test_boilerplate_store_score_lower_than_clinical(self) -> None:
        clinical = _make_chunk(CLINICAL_TEXT)
        boilerplate = _make_chunk(BOILERPLATE_TEXT)
        r_clinical = score_chunk_moe(clinical)
        r_boilerplate = score_chunk_moe(boilerplate)
        assert r_clinical.store_score > r_boilerplate.store_score


# ===================================================================
# 7. Ordering (2 tests)
# ===================================================================

class TestOrdering:
    """Domain-important content scores higher than generic content."""

    def test_clinical_higher_than_boilerplate(self) -> None:
        clinical = _make_chunk(CLINICAL_TEXT)
        boilerplate = _make_chunk(BOILERPLATE_TEXT)
        score_chunk_moe(clinical)
        score_chunk_moe(boilerplate)
        assert clinical.metadata["moe_store_score"] > boilerplate.metadata["moe_store_score"]

    def test_novel_higher_than_generic(self) -> None:
        novel = _make_chunk(NOVEL_TEXT)
        generic = _make_chunk("The sky is blue and water is wet.")
        score_chunk_moe(novel)
        score_chunk_moe(generic)
        assert novel.metadata["moe_combined_score"] > generic.metadata["moe_combined_score"]


# ===================================================================
# 8. MoEResult.to_dict() (1 test)
# ===================================================================

class TestMoEResultToDict:
    """Serialisation method produces the expected shape."""

    def test_to_dict_structure(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        d = result.to_dict()
        assert "combined_score" in d
        assert "store_score" in d
        assert "votes" in d
        assert isinstance(d["votes"], dict)
        # votes should be name -> float (not ExpertVote)
        for name, val in d["votes"].items():
            assert isinstance(name, str)
            assert isinstance(val, float)


# ===================================================================
# 9. ExpertVote frozen (1 test)
# ===================================================================

class TestExpertVoteFrozen:
    """ExpertVote dataclass is immutable."""

    def test_cannot_mutate_expert_vote(self) -> None:
        vote = ExpertVote(name="test", weight=1.0, score=0.5, hits=3)
        with pytest.raises(AttributeError):
            vote.score = 0.9  # type: ignore[misc]


# ===================================================================
# 10. score_chunks_moe batch (2 tests)
# ===================================================================

class TestScoreChunksMoeBatch:
    """Batch scoring processes a list and returns the same list."""

    def test_returns_same_list_object(self) -> None:
        chunks = [_make_chunk(CLINICAL_TEXT), _make_chunk(BOILERPLATE_TEXT)]
        returned = score_chunks_moe(chunks)
        assert returned is chunks

    def test_all_chunks_scored(self) -> None:
        chunks = [
            _make_chunk(CLINICAL_TEXT),
            _make_chunk(BOILERPLATE_TEXT),
            _make_chunk(NOVEL_TEXT),
        ]
        score_chunks_moe(chunks)
        for chunk in chunks:
            assert "moe_combined_score" in chunk.metadata
            assert "moe_store_score" in chunk.metadata
            assert "moe_votes" in chunk.metadata


# ===================================================================
# 11. Saturation cap (1 test)
# ===================================================================

class TestSaturationCap:
    """More than 5 keyword hits still caps the expert score at 1.0."""

    def test_score_capped_at_one(self) -> None:
        # 8 reproducibility keywords in one text -> hits > 5
        text = (
            "import basic boilerplate template default standard trivial generic "
            "import basic boilerplate template"
        )
        tokens = _tokenize(text)
        expert = next(e for e in _EXPERTS if e.name == "reproducibility")
        score, hits = _score_expert(expert, tokens)
        assert hits > 5
        assert score == 1.0


# ===================================================================
# 12. Mixed content (1 test)
# ===================================================================

class TestMixedContent:
    """Text with keywords from multiple experts activates several votes."""

    def test_mixed_keywords_multiple_experts(self) -> None:
        text = (
            "This novel protocol must ensure safety compliance. "
            "The algorithm provides actionable feedback with evidence."
        )
        chunk = _make_chunk(text)
        result = score_chunk_moe(chunk)
        # At least accuracy_criticality, novelty, methodology, usability,
        # context_sovereignty, and emergence_leverage should have hits
        active = [n for n, v in result.votes.items() if v.hits > 0]
        assert len(active) >= 4


# ===================================================================
# 13. Complexity score integration (2 tests)
# ===================================================================

class TestComplexityScoreIntegration:
    """chunk.metadata['complexity_score'] influences the store score."""

    def test_higher_complexity_raises_store_score(self) -> None:
        chunk_low = _make_chunk(CLINICAL_TEXT, complexity_score=0.0)
        chunk_high = _make_chunk(CLINICAL_TEXT, complexity_score=1.0)
        r_low = score_chunk_moe(chunk_low)
        r_high = score_chunk_moe(chunk_high)
        # complexity coefficient is +0.10, so high should be higher
        assert r_high.store_score > r_low.store_score

    def test_default_complexity_is_zero(self) -> None:
        chunk = _make_chunk(CLINICAL_TEXT)
        result = score_chunk_moe(chunk)
        # Verify the store score equals the formula with complexity_score=0
        v = result.votes
        expected = (
            0.25 * v["accuracy_criticality"].score
            + 0.20 * v["novelty"].score
            + 0.20 * v["methodology"].score
            + 0.15 * v["emergence_leverage"].score
            + 0.10 * v["context_sovereignty"].score
            - 0.30 * v["reproducibility"].score
            + 0.10 * 0.0
        )
        assert abs(result.store_score - round(expected, 6)) < 1e-6


# ===================================================================
# 14. Tokenizer (1 test)
# ===================================================================

class TestTokenizer:
    """_tokenize produces expected lowercase alphanumeric tokens."""

    def test_tokenize_strips_punctuation_and_lowercases(self) -> None:
        tokens = _tokenize("Hello, World! 42 is the answer.")
        assert tokens == ["hello", "world", "42", "is", "the", "answer"]
