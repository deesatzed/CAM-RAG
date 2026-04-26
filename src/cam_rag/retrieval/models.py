"""Small retrieval result containers shared by retrieval primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    """Document accepted by the lightweight sparse index."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Generic ranked retrieval result."""

    doc_id: str
    text: str
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FusedResult:
    """Document-level result after Reciprocal Rank Fusion."""

    doc_id: str
    rrf_score: float
    rank: int
    source_ranks: dict[str, int]
    source_scores: dict[str, float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
