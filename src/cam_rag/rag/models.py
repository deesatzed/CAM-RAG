"""Shared platform models for document, methodology, and graph RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

ChunkLevel = Literal["sentence", "paragraph", "section", "document", "methodology"]


class EvidenceSignals(TypedDict, total=False):
    """Known keys produced by hybrid RRF retrieval.

    All keys are optional (total=False) because Evidence can be constructed
    with partial signals by downstream consumers or tests.
    """

    matched_terms: list[str]
    source_ranks: dict[str, int]
    source_scores: dict[str, float]
    rrf_score: float


class RetrievalStats(TypedDict, total=False):
    """Known keys populated in RAGTrace.retrieval_stats by the query path.

    All keys are optional (total=False) because the trace may be partially
    populated depending on the code path (e.g. query_expansion may not run).
    """

    documents: int
    chunks: int
    evidence: int
    expanded_query: str


@dataclass(slots=True)
class CorpusDocument:
    """Normalized source document before chunking or indexing."""

    id: str
    text: str
    source: str
    title: str = ""
    format: str = "text"
    # metadata is genuinely dynamic: user-supplied key/value pairs that vary
    # per document format and ingestion pipeline.  dict[str, Any] is correct.
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
    # metadata is genuinely dynamic: user-supplied key/value pairs that vary
    # per chunk and ingestion pipeline.  dict[str, Any] is correct.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Evidence:
    """Ranked retrieval result with explainable scoring signals."""

    chunk: Chunk
    score: float
    retriever: str
    rank: int = 0
    signals: EvidenceSignals = field(default_factory=dict)  # type: ignore[assignment]
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
    retrieval_stats: RetrievalStats = field(default_factory=dict)  # type: ignore[assignment]
    # confidence_details merges ConfidenceReport.to_dict() with a nested
    # "grounding" key from GroundingReport.to_dict().  The shape is a
    # heterogeneous mix of floats, ints, and a nested dict produced by
    # dataclasses.asdict(), so dict[str, Any] is the most accurate type here.
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
