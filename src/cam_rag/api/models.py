"""Pydantic request/response models for the CAM RAG API.

The domain models (RAGAnswer, Evidence, etc.) are plain dataclasses with
``slots=True``.  This module provides Pydantic wrappers that handle
JSON serialization, request validation, and dataclass-to-dict conversion.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class DocumentIn(BaseModel):
    """Inline document supplied with a query request."""

    id: str
    text: str
    source: str
    title: str = ""


class ConfigOverrides(BaseModel):
    """Optional spec overrides the caller may supply per request.

    Only serializable, commonly-toggled fields are exposed.  Backend objects
    (generation_backend, reranker_backend, embedding_backend) are not
    settable via the API -- they must be configured server-side.
    """

    retrieval_top_k: int | None = None
    dense_weight: float | None = None
    sparse_weight: float | None = None
    chunk_overlap: int | None = None
    query_expansion_enabled: bool | None = None
    expansion_terms: int | None = None
    use_pipeline: bool | None = None
    use_rl: bool | None = None
    use_governance: bool | None = None
    enforce_phi: bool | None = None
    enforce_pii: bool | None = None
    require_citations: bool | None = None
    min_confidence: float | None = None


class QueryRequest(BaseModel):
    """POST /v1/query request body."""

    question: str
    documents: list[DocumentIn]
    config: ConfigOverrides | None = None


class QueryFolderRequest(BaseModel):
    """POST /v1/query-folder request body."""

    question: str
    docs_dir: str
    config: ConfigOverrides | None = None
    limit: int | None = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ChunkOut(BaseModel):
    """Serialized Chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    text: str
    level: str = "paragraph"
    source: str = ""
    title: str = ""
    parent_id: str | None = None
    position: int = 0
    page_number: int | None = None
    section_heading: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceOut(BaseModel):
    """Serialized Evidence."""

    model_config = ConfigDict(from_attributes=True)

    chunk: ChunkOut
    score: float
    retriever: str
    rank: int = 0
    signals: dict[str, Any] = Field(default_factory=dict)
    graph_reasons: list[str] = Field(default_factory=list)


class CitationOut(BaseModel):
    """Serialized Citation."""

    model_config = ConfigDict(from_attributes=True)

    source: str
    document_id: str
    title: str = ""
    page_number: int | None = None
    section_heading: str = ""
    excerpt: str = ""
    score: float = 0.0


class TraceOut(BaseModel):
    """Serialized RAGTrace."""

    model_config = ConfigDict(from_attributes=True)

    query_type: str = "unknown"
    stages: list[str] = Field(default_factory=list)
    retrieval_stats: dict[str, Any] = Field(default_factory=dict)
    confidence_details: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response body for /v1/query and /v1/query-folder.

    Mirrors the RAGAnswer dataclass.
    """

    model_config = ConfigDict(from_attributes=True)

    answer: str
    evidence: list[EvidenceOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: float = 0.0
    grounded: bool = False
    trace: TraceOut = Field(default_factory=TraceOut)


class HealthResponse(BaseModel):
    """GET /v1/health response."""

    status: str
    version: str


class ConfigResponse(BaseModel):
    """GET /v1/config response -- available feature flags and defaults."""

    feature_flags: dict[str, Any]
    policy_defaults: dict[str, Any]
