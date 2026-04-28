"""Methodology-family data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


@dataclass(slots=True)
class Methodology:
    problem: str
    solution: str
    language: str = "unknown"
    tags: list[str] = field(default_factory=list)
    source: str = ""
    notes: str = ""
    id: str = field(default_factory=lambda: new_id("meth"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Methodology:
        return cls(**data)


@dataclass(slots=True)
class Family:
    name: str
    barcode: str
    status: str = "approved"
    category: str = ""
    language: str = "unknown"
    source_repos: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("fam"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Family:
        return cls(**data)


@dataclass(slots=True)
class Membership:
    family_id: str
    methodology_id: str
    confidence: float = 1.0
    review_status: str = "approved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Membership:
        return cls(**data)


@dataclass(slots=True)
class RowResult:
    methodology: Methodology
    score: float
    source: str = "lexical"


@dataclass(slots=True)
class FamilyResult:
    key: str
    score: float
    family: Family | None
    representative: Methodology
    methodology_ids: list[str]
    source: str = "direct"
    graph_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": round(self.score, 6),
            "source": self.source,
            "graph_reasons": self.graph_reasons,
            "family": self.family.to_dict() if self.family else None,
            "representative": self.representative.to_dict(),
            "methodology_ids": self.methodology_ids,
        }
