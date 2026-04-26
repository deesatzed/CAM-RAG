"""Shared platform models for document, methodology, and graph RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ChunkLevel = Literal["sentence", "paragraph", "section", "document", "methodology"]


@dataclass(slots=True)
class CorpusDocument:
    """Normalized source document before chunking or indexing."""

    id: str
    text: str
    source: str
    title: str = ""
    format: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Normalized retrievable unit."""

    id: str
    document_id: str
    text: str
    level: ChunkLevel = "paragraph"
    source: str = ""
    title: str = ""
    parent_id: str | None = None
    position: int = 0
    page_number: int | None = None
    section_heading: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Evidence:
    """Ranked retrieval result with explainable scoring signals."""

    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0
    signals: dict[str, Any] = field(default_factory=dict)
    graph_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    """Citation emitted with an answer."""

    source: str
    document_id: str
    title: str = ""
    page_number: int | None = None
    section_heading: str = ""
    excerpt: str = ""
    score: float = 0.0


@dataclass(slots=True)
class RAGTrace:
    """Structured trace for observability and benchmarking."""

    query_type: str = "unknown"
    stages: list[str] = field(default_factory=list)
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    confidence_details: dict[str, Any] = field(default_factory=dict)

    def add(self, stage: str) -> None:
        self.stages.append(stage)


@dataclass(slots=True)
class RAGAnswer:
    """Final platform response."""

    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    trace: RAGTrace = field(default_factory=RAGTrace)
