"""Citation grounding checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cam_rag.rag.models import Citation, Evidence


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Result of checking citations against retrieved evidence."""

    grounded: bool
    checked_citations: int
    matched_citations: int
    missing_sources: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def verify_citations_grounded(
    citations: list[Citation],
    evidence: list[Evidence],
    *,
    require_citations: bool = True,
) -> GroundingReport:
    """Verify that answer citations correspond to retrieved evidence chunks."""

    if not citations:
        return GroundingReport(
            grounded=not require_citations,
            checked_citations=0,
            matched_citations=0,
            missing_sources=[],
        )

    evidence_keys = {
        (item.chunk.document_id, item.chunk.source, item.chunk.position)
        for item in evidence
    }
    matched = 0
    missing_sources: list[str] = []
    for citation in citations:
        key_prefix = (citation.document_id, citation.source)
        found = any(
            document_id == key_prefix[0] and source == key_prefix[1]
            for document_id, source, _position in evidence_keys
        )
        if found:
            matched += 1
        else:
            missing_sources.append(citation.source)

    return GroundingReport(
        grounded=matched == len(citations),
        checked_citations=len(citations),
        matched_citations=matched,
        missing_sources=missing_sources,
    )
