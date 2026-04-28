"""Tests for ETF (Epistemic Tension Field) verification."""

import pytest

from cam_rag.rag.models import Chunk, Evidence
from cam_rag.verification.etf import (
    ETFReport,
    _GENERIC_PHRASES,
    _STOP_WORDS,
    compute_etf,
    compute_skeptic,
    compute_sovereignty,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_evidence(
    text: str, doc_id: str = "doc1", chunk_id: str = "c1"
) -> Evidence:
    return Evidence(
        chunk=Chunk(id=chunk_id, document_id=doc_id, text=text),
        score=0.8,
        retriever="test",
    )


# ---------------------------------------------------------------------------
# 1. High evidence overlap -> high sovereignty (2 tests)
# ---------------------------------------------------------------------------


class TestHighSovereignty:
    def test_identical_text_yields_sovereignty_one(self) -> None:
        """When answer repeats evidence verbatim, sovereignty should be 1.0."""
        text = "Metformin reduces hepatic glucose production effectively"
        evidence = [_make_evidence(text)]
        ratio, used, total = compute_sovereignty(text, evidence)
        assert ratio == 1.0
        assert used == total
        assert total > 0

    def test_near_complete_overlap(self) -> None:
        """Most answer terms sourced from evidence yields high sovereignty."""
        evidence_text = (
            "Metformin is the first-line treatment for type 2 diabetes mellitus. "
            "It works by reducing hepatic glucose production and improving insulin sensitivity."
        )
        answer = (
            "Metformin is the first-line treatment for type 2 diabetes. "
            "It reduces hepatic glucose production."
        )
        evidence = [_make_evidence(evidence_text)]
        ratio, used, total = compute_sovereignty(answer, evidence)
        # All content words in answer should appear in evidence
        assert ratio >= 0.8
        assert used >= 5


# ---------------------------------------------------------------------------
# 2. No overlap -> low sovereignty (1 test)
# ---------------------------------------------------------------------------


class TestNoOverlapSovereignty:
    def test_completely_disjoint_text(self) -> None:
        """When answer and evidence share zero content words, ratio is 0.0."""
        evidence = [_make_evidence("Aspirin inhibits platelet aggregation")]
        answer = "Quantum entanglement produces nonlocal correlations"
        ratio, used, total = compute_sovereignty(answer, evidence)
        assert ratio == 0.0
        assert used == 0
        assert total > 0


# ---------------------------------------------------------------------------
# 3. Partial overlap (1 test)
# ---------------------------------------------------------------------------


class TestPartialOverlap:
    def test_partial_term_overlap(self) -> None:
        """Some shared, some novel terms yields a mid-range ratio."""
        evidence = [_make_evidence("Insulin sensitivity improves with exercise")]
        answer = "Insulin sensitivity may also improve with dietary changes"
        ratio, used, total = compute_sovereignty(answer, evidence)
        assert 0.0 < ratio < 1.0
        assert used >= 2  # at least "insulin", "sensitivity"


# ---------------------------------------------------------------------------
# 4. No generic phrases -> skeptic 1.0 (1 test)
# ---------------------------------------------------------------------------


class TestCleanSkeptic:
    def test_no_generics_yields_perfect_skeptic(self) -> None:
        answer = "Metformin reduces hepatic glucose production by 25 percent"
        score, count, found = compute_skeptic(answer)
        assert score == 1.0
        assert count == 0
        assert found == []


# ---------------------------------------------------------------------------
# 5. Loaded with generics -> low skeptic (2 tests)
# ---------------------------------------------------------------------------


class TestGenericSkeptic:
    def test_single_generic_phrase(self) -> None:
        answer = "In general, the drug is effective"
        score, count, found = compute_skeptic(answer)
        assert score == pytest.approx(0.85)
        assert count == 1
        assert found == ["in general"]

    def test_many_generic_phrases(self) -> None:
        answer = (
            "In general, it is well known that, broadly speaking, "
            "as a general rule, the treatment is typically speaking effective. "
            "It goes without saying that needless to say results vary."
        )
        score, count, found = compute_skeptic(answer)
        assert count >= 6
        # 6 * 0.15 = 0.90, so score = max(0.0, 1.0 - 0.90) = 0.10
        assert score <= 0.25


# ---------------------------------------------------------------------------
# 6. Empty answer (1 test)
# ---------------------------------------------------------------------------


class TestEmptyAnswer:
    def test_empty_answer_returns_safe_defaults(self) -> None:
        evidence = [_make_evidence("Some evidence text here")]
        report = compute_etf("", evidence)
        assert report.sovereignty_score == 0.0
        assert report.grounding_ratio == 0.0
        assert report.answer_terms == 0
        assert report.evidence_terms_used == 0
        # skeptic should be clean since no text has no generic phrases
        assert report.skeptic_score == 1.0
        assert report.generic_count == 0


# ---------------------------------------------------------------------------
# 7. Empty evidence (1 test)
# ---------------------------------------------------------------------------


class TestEmptyEvidence:
    def test_empty_evidence_yields_zero_sovereignty(self) -> None:
        answer = "Metformin reduces glucose production"
        report = compute_etf(answer, [])
        assert report.sovereignty_score == 0.0
        assert report.grounding_ratio == 0.0
        assert report.evidence_terms_used == 0
        # answer_terms should still be counted
        assert report.answer_terms > 0


# ---------------------------------------------------------------------------
# 8. Overall score weighting (2 tests)
# ---------------------------------------------------------------------------


class TestOverallWeighting:
    def test_default_weights_60_40(self) -> None:
        """Verify default 0.6 sovereignty + 0.4 skeptic weighting."""
        evidence_text = "Alpha bravo charlie"
        answer = "Alpha bravo charlie"  # perfect sovereignty
        evidence = [_make_evidence(evidence_text)]
        report = compute_etf(answer, evidence)
        # sovereignty = 1.0, skeptic = 1.0
        expected = 0.6 * 1.0 + 0.4 * 1.0
        assert report.overall_score == pytest.approx(expected)

    def test_custom_weights(self) -> None:
        """Caller can override sovereignty/skeptic weights."""
        evidence_text = "Alpha bravo charlie"
        answer = "Alpha bravo charlie"
        evidence = [_make_evidence(evidence_text)]
        report = compute_etf(
            answer, evidence, sovereignty_weight=0.3, skeptic_weight=0.7
        )
        expected = 0.3 * 1.0 + 0.7 * 1.0
        assert report.overall_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 9. to_dict() (1 test)
# ---------------------------------------------------------------------------


class TestToDict:
    def test_all_fields_present(self) -> None:
        evidence = [_make_evidence("Sample evidence text")]
        report = compute_etf("Sample answer text", evidence)
        d = report.to_dict()
        expected_keys = {
            "sovereignty_score",
            "skeptic_score",
            "overall_score",
            "grounding_ratio",
            "evidence_terms_used",
            "answer_terms",
            "generic_count",
            "generic_phrases_found",
        }
        assert set(d.keys()) == expected_keys
        # generic_phrases_found should be serialized as list, not tuple
        assert isinstance(d["generic_phrases_found"], list)


# ---------------------------------------------------------------------------
# 10. ETFReport frozen (1 test)
# ---------------------------------------------------------------------------


class TestFrozenReport:
    def test_cannot_mutate_report(self) -> None:
        evidence = [_make_evidence("text")]
        report = compute_etf("text", evidence)
        with pytest.raises(AttributeError):
            report.sovereignty_score = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 11. Real clinical text (2 tests)
# ---------------------------------------------------------------------------


class TestClinicalText:
    def test_clinical_evidence_and_answer(self) -> None:
        evidence_text = (
            "Metformin is the first-line treatment for type 2 diabetes mellitus. "
            "It works by reducing hepatic glucose production and improving insulin sensitivity."
        )
        answer = (
            "According to the evidence, metformin is the first-line treatment "
            "for type 2 diabetes. It reduces hepatic glucose production."
        )
        evidence = [_make_evidence(evidence_text)]
        report = compute_etf(answer, evidence)
        # Answer is well grounded in clinical evidence
        assert report.sovereignty_score >= 0.6
        assert report.skeptic_score == 1.0
        assert report.overall_score > 0.5

    def test_clinical_with_boilerplate(self) -> None:
        evidence_text = (
            "Lisinopril reduces blood pressure by inhibiting angiotensin converting enzyme."
        )
        answer = (
            "In general, it is well known that lisinopril reduces blood pressure. "
            "Broadly speaking, ACE inhibitors are effective."
        )
        evidence = [_make_evidence(evidence_text)]
        report = compute_etf(answer, evidence)
        # Has generic phrases so skeptic should be penalized
        assert report.generic_count >= 3
        assert report.skeptic_score < 1.0
        # Still has some grounding
        assert report.sovereignty_score > 0.0


# ---------------------------------------------------------------------------
# 12. Stop words excluded (1 test)
# ---------------------------------------------------------------------------


class TestStopWordExclusion:
    def test_stop_words_not_counted_as_terms(self) -> None:
        """An answer of only stop words should yield 0 answer_terms."""
        answer = "the is a an are was were"
        evidence = [_make_evidence("the is a an are was were")]
        ratio, used, total = compute_sovereignty(answer, evidence)
        assert total == 0
        assert used == 0
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# 13. Case insensitivity (1 test)
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    def test_tokens_matched_case_insensitively(self) -> None:
        evidence = [_make_evidence("METFORMIN REDUCES GLUCOSE")]
        answer = "metformin reduces glucose"
        ratio, used, total = compute_sovereignty(answer, evidence)
        assert ratio == 1.0
        assert used == total


# ---------------------------------------------------------------------------
# 14. Generic phrases case insensitive (1 test)
# ---------------------------------------------------------------------------


class TestGenericCaseInsensitive:
    def test_uppercase_generic_still_detected(self) -> None:
        answer = "IN GENERAL the treatment works. IT IS WELL KNOWN to be effective."
        score, count, found = compute_skeptic(answer)
        assert count == 2
        assert "in general" in found
        assert "it is well known" in found


# ---------------------------------------------------------------------------
# 15. Skeptic floor at 0.0 (1 test)
# ---------------------------------------------------------------------------


class TestSkepticFloor:
    def test_many_generics_do_not_go_below_zero(self) -> None:
        """Even with 10+ matches the score floors at 0.0."""
        phrases = list(_GENERIC_PHRASES[:10])
        answer = ". ".join(phrases)
        score, count, found = compute_skeptic(answer)
        assert count >= 10
        # 10 * 0.15 = 1.5 => max(0.0, 1.0 - 1.5) = 0.0
        assert score == 0.0


# ---------------------------------------------------------------------------
# 16. Multiple evidence items (2 tests)
# ---------------------------------------------------------------------------


class TestMultipleEvidence:
    def test_terms_from_all_evidence_combined(self) -> None:
        """Sovereignty unions terms across all evidence chunks."""
        ev1 = _make_evidence("Alpha bravo", chunk_id="c1")
        ev2 = _make_evidence("Charlie delta", chunk_id="c2")
        answer = "Alpha bravo charlie delta"
        ratio, used, total = compute_sovereignty(answer, [ev1, ev2])
        assert ratio == 1.0
        assert used == 4

    def test_partial_coverage_across_evidence(self) -> None:
        ev1 = _make_evidence("Alpha bravo", chunk_id="c1")
        ev2 = _make_evidence("Charlie delta", chunk_id="c2")
        answer = "Alpha charlie echo foxtrot"
        ratio, used, total = compute_sovereignty(answer, [ev1, ev2])
        assert used == 2  # alpha, charlie
        assert total == 4
        assert ratio == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 17. Answer with all evidence terms -> sovereignty near 1.0 (1 test)
# ---------------------------------------------------------------------------


class TestFullEvidence:
    def test_answer_covers_all_evidence_terms(self) -> None:
        evidence_text = "Penicillin treats streptococcal pharyngitis"
        answer = "Penicillin treats streptococcal pharyngitis effectively"
        evidence = [_make_evidence(evidence_text)]
        ratio, used, total = compute_sovereignty(answer, evidence)
        # "effectively" is extra, but all evidence terms appear in answer
        # ratio = 4/5 = 0.8 (4 shared out of 5 answer terms)
        assert ratio >= 0.75
        assert used >= 4


# ---------------------------------------------------------------------------
# Constant sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_stop_words_count(self) -> None:
        """Stop words set has the expected size (approximately 40)."""
        assert len(_STOP_WORDS) >= 30

    def test_generic_phrases_count(self) -> None:
        """Generic phrases tuple has the expected size (20)."""
        assert len(_GENERIC_PHRASES) == 20
