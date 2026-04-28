"""Epistemic Tension Field (ETF) verification.

Two heuristic dimensions — Context Sovereignty and Model Prior Skeptic —
are combined into a single ETF score that measures how well an answer is
grounded in retrieved evidence and how free it is from generic boilerplate
language.  No LLM calls are made; all scoring is deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cam_rag.rag.models import Evidence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "it",
        "this",
        "that",
    }
)

_GENERIC_PHRASES: tuple[str, ...] = (
    "in general",
    "it is well known",
    "as a general rule",
    "typically speaking",
    "broadly speaking",
    "it is commonly accepted",
    "as we all know",
    "in most cases",
    "generally speaking",
    "it goes without saying",
    "needless to say",
    "as a matter of fact",
    "for all intents and purposes",
    "at the end of the day",
    "by and large",
    "all things considered",
    "in the grand scheme",
    "more or less",
    "as such",
    "it should be noted",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ETFReport:
    """Immutable result of an Epistemic Tension Field evaluation."""

    sovereignty_score: float
    skeptic_score: float
    overall_score: float
    grounding_ratio: float
    evidence_terms_used: int
    answer_terms: int
    generic_count: int
    generic_phrases_found: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "sovereignty_score": self.sovereignty_score,
            "skeptic_score": self.skeptic_score,
            "overall_score": self.overall_score,
            "grounding_ratio": self.grounding_ratio,
            "evidence_terms_used": self.evidence_terms_used,
            "answer_terms": self.answer_terms,
            "generic_count": self.generic_count,
            "generic_phrases_found": list(self.generic_phrases_found),
        }


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Return a set of lowercase tokens with stop words removed."""
    return {
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text)
        if m.group(0).lower() not in _STOP_WORDS
    }


# ---------------------------------------------------------------------------
# Dimension 1 — Context Sovereignty
# ---------------------------------------------------------------------------


def compute_sovereignty(
    answer: str, evidence: list[Evidence]
) -> tuple[float, int, int]:
    """Measure how well the answer is grounded in evidence terms.

    Returns
    -------
    tuple[float, int, int]
        (grounding_ratio, evidence_terms_used_in_answer, total_answer_terms)
    """
    answer_terms = _tokenize(answer)
    if not answer_terms:
        return (0.0, 0, 0)

    evidence_terms: set[str] = set()
    for item in evidence:
        evidence_terms |= _tokenize(item.chunk.text)

    overlap = answer_terms & evidence_terms
    grounding_ratio = len(overlap) / len(answer_terms)
    return (grounding_ratio, len(overlap), len(answer_terms))


# ---------------------------------------------------------------------------
# Dimension 2 — Model Prior Skeptic
# ---------------------------------------------------------------------------


def compute_skeptic(answer: str) -> tuple[float, int, list[str]]:
    """Detect generic / boilerplate language in the answer.

    Returns
    -------
    tuple[float, int, list[str]]
        (skeptic_score, generic_count, list_of_matched_phrases)
    """
    lower_answer = answer.lower()
    found: list[str] = []
    for phrase in _GENERIC_PHRASES:
        if phrase in lower_answer:
            found.append(phrase)

    generic_count = len(found)
    skeptic_score = max(0.0, 1.0 - (generic_count * 0.15))
    return (skeptic_score, generic_count, found)


# ---------------------------------------------------------------------------
# Combined ETF score
# ---------------------------------------------------------------------------


def compute_etf(
    answer: str,
    evidence: list[Evidence],
    *,
    sovereignty_weight: float = 0.6,
    skeptic_weight: float = 0.4,
) -> ETFReport:
    """Compute the full Epistemic Tension Field report.

    Parameters
    ----------
    answer:
        The generated answer text.
    evidence:
        The retrieved evidence items that informed the answer.
    sovereignty_weight:
        Weight for the context sovereignty dimension (default 0.6).
    skeptic_weight:
        Weight for the model prior skeptic dimension (default 0.4).

    Returns
    -------
    ETFReport
        Frozen dataclass with all scoring details.
    """
    grounding_ratio, evidence_terms_used, answer_term_count = compute_sovereignty(
        answer, evidence
    )
    skeptic_score, generic_count, generic_phrases_found = compute_skeptic(answer)

    sovereignty_score = grounding_ratio
    overall = sovereignty_weight * sovereignty_score + skeptic_weight * skeptic_score

    return ETFReport(
        sovereignty_score=sovereignty_score,
        skeptic_score=skeptic_score,
        overall_score=round(overall, 6),
        grounding_ratio=grounding_ratio,
        evidence_terms_used=evidence_terms_used,
        answer_terms=answer_term_count,
        generic_count=generic_count,
        generic_phrases_found=tuple(generic_phrases_found),
    )
