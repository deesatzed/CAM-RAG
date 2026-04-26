"""Retrieval confidence scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cam_rag.rag.models import Evidence


@dataclass(frozen=True, slots=True)
class ConfidenceReport:
    """Explainable confidence summary for a retrieval result set."""

    overall: float
    top_score: float
    score_margin: float
    evidence_count: int
    source_agreement: float
    query_coverage: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def score_retrieval_confidence(evidence: list[Evidence]) -> ConfidenceReport:
    """Compute bounded confidence from retrieval signals.

    The score is intentionally conservative. It rewards a strong top result,
    agreement between dense/sparse sources, query term coverage, and multiple
    evidence items.
    """

    if not evidence:
        return ConfidenceReport(
            overall=0.0,
            top_score=0.0,
            score_margin=0.0,
            evidence_count=0,
            source_agreement=0.0,
            query_coverage=0.0,
        )

    top_score = max(0.0, evidence[0].score)
    second_score = max(0.0, evidence[1].score) if len(evidence) > 1 else 0.0
    score_margin = max(0.0, top_score - second_score)
    source_agreement = _source_agreement(evidence[0])
    query_coverage = _query_coverage(evidence[0])
    evidence_count_score = min(1.0, len(evidence) / 3)

    # RRF scores are small, so top_score is treated as a relative signal and
    # capped at a low scale. Coverage/agreement carry more weight.
    normalized_top = min(1.0, top_score * 80)
    normalized_margin = min(1.0, score_margin * 120)
    overall = (
        normalized_top * 0.25
        + normalized_margin * 0.10
        + source_agreement * 0.25
        + query_coverage * 0.30
        + evidence_count_score * 0.10
    )
    return ConfidenceReport(
        overall=round(max(0.0, min(1.0, overall)), 6),
        top_score=top_score,
        score_margin=score_margin,
        evidence_count=len(evidence),
        source_agreement=source_agreement,
        query_coverage=query_coverage,
    )


def _source_agreement(evidence: Evidence) -> float:
    ranks = evidence.signals.get("source_ranks", {})
    if not isinstance(ranks, dict) or not ranks:
        return 0.0
    if len(ranks) >= 2:
        return 1.0
    return 0.5


def _query_coverage(evidence: Evidence) -> float:
    matched_terms = evidence.signals.get("matched_terms", [])
    if not isinstance(matched_terms, list):
        return 0.0
    source_scores = evidence.signals.get("source_scores", {})
    if isinstance(source_scores, dict) and "sparse" in source_scores:
        # Query term count is not stored here, so use coverage-like saturation
        # from matched term count. Exact coverage can be added by retrievers.
        return min(1.0, len(matched_terms) / 4)
    return min(1.0, len(matched_terms) / 5)
