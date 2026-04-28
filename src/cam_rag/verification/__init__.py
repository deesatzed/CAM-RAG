"""Verification and confidence helpers for RAG answers."""

from cam_rag.verification.confidence import ConfidenceReport, score_retrieval_confidence
from cam_rag.verification.etf import ETFReport, compute_etf, compute_skeptic, compute_sovereignty
from cam_rag.verification.grounding import GroundingReport, verify_citations_grounded
from cam_rag.verification.sensitive import SensitiveMatch, redact_text, scan_text

__all__ = [
    "ConfidenceReport",
    "ETFReport",
    "GroundingReport",
    "SensitiveMatch",
    "compute_etf",
    "compute_skeptic",
    "compute_sovereignty",
    "redact_text",
    "scan_text",
    "score_retrieval_confidence",
    "verify_citations_grounded",
]
