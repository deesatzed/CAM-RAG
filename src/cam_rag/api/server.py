"""FastAPI server exposing the CAM RAG platform query functions as HTTP endpoints.

Run with::

    uvicorn cam_rag.api.server:app --reload
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException

from cam_rag.api.models import (
    ConfigOverrides,
    ConfigResponse,
    HealthResponse,
    QueryFolderRequest,
    QueryRequest,
    QueryResponse,
)
from cam_rag.query import query as platform_query
from cam_rag.query import query_document_folder as platform_query_folder
from cam_rag.rag.models import CorpusDocument, RAGAnswer
from cam_rag.rag.spec import RAGAppSpec, RAGPolicy

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"

app = FastAPI(
    title="CAM RAG Platform API",
    description="HTTP/REST API for the CAM RAG retrieval-augmented generation platform.",
    version=_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_overrides(spec: RAGAppSpec, overrides: ConfigOverrides | None) -> RAGAppSpec:
    """Apply caller-supplied config overrides to a fresh RAGAppSpec.

    Only non-None fields in *overrides* are applied.  Backend objects are
    intentionally excluded -- they must be configured server-side.
    """
    if overrides is None:
        return spec

    spec_fields = {
        "retrieval_top_k",
        "dense_weight",
        "sparse_weight",
        "chunk_overlap",
        "query_expansion_enabled",
        "expansion_terms",
        "use_pipeline",
        "use_rl",
        "use_governance",
        "moe_scoring_enabled",
        "accuracy_contracts_enabled",
        "selective_filter_enabled",
        "selective_filter_threshold",
        "etf_verification_enabled",
        "complexity_routing_enabled",
    }
    policy_fields = {
        "enforce_phi",
        "enforce_pii",
        "require_citations",
        "min_confidence",
    }

    for field_name in spec_fields:
        value = getattr(overrides, field_name, None)
        if value is not None:
            object.__setattr__(spec, field_name, value)

    for field_name in policy_fields:
        value = getattr(overrides, field_name, None)
        if value is not None:
            object.__setattr__(spec.policy, field_name, value)

    return spec


def _rag_answer_to_response(answer: RAGAnswer) -> QueryResponse:
    """Convert a domain RAGAnswer dataclass to the Pydantic response model."""
    return QueryResponse.model_validate(asdict(answer))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", version=_VERSION)


@app.get("/v1/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    """Return available feature flags and their default values."""
    default_spec = RAGAppSpec(name="default")
    default_policy = RAGPolicy()

    feature_flags = {
        "retrieval_top_k": default_spec.retrieval_top_k,
        "dense_weight": default_spec.dense_weight,
        "sparse_weight": default_spec.sparse_weight,
        "chunk_overlap": default_spec.chunk_overlap,
        "query_expansion_enabled": default_spec.query_expansion_enabled,
        "expansion_terms": default_spec.expansion_terms,
        "use_pipeline": default_spec.use_pipeline,
        "use_rl": default_spec.use_rl,
        "use_governance": default_spec.use_governance,
        "moe_scoring_enabled": default_spec.moe_scoring_enabled,
        "accuracy_contracts_enabled": default_spec.accuracy_contracts_enabled,
        "selective_filter_enabled": default_spec.selective_filter_enabled,
        "selective_filter_threshold": default_spec.selective_filter_threshold,
        "etf_verification_enabled": default_spec.etf_verification_enabled,
        "complexity_routing_enabled": default_spec.complexity_routing_enabled,
    }
    policy_defaults = {
        "enforce_phi": default_policy.enforce_phi,
        "enforce_pii": default_policy.enforce_pii,
        "require_citations": default_policy.require_citations,
        "min_confidence": default_policy.min_confidence,
        "use_llm_redaction": default_policy.use_llm_redaction,
    }
    return ConfigResponse(feature_flags=feature_flags, policy_defaults=policy_defaults)


@app.post("/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query inline documents and return a cited RAG answer."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents list must not be empty")

    corpus_docs = [
        CorpusDocument(
            id=doc.id,
            text=doc.text,
            source=doc.source,
            title=doc.title,
        )
        for doc in request.documents
    ]

    spec = RAGAppSpec(name="api")
    spec = _apply_overrides(spec, request.config)

    try:
        answer = platform_query(
            question=request.question,
            documents=corpus_docs,
            spec=spec,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("query failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return _rag_answer_to_response(answer)


@app.post("/v1/query-folder", response_model=QueryResponse)
async def query_folder(request: QueryFolderRequest) -> QueryResponse:
    """Load a document folder, retrieve evidence, and return a cited RAG answer."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if not request.docs_dir or not request.docs_dir.strip():
        raise HTTPException(status_code=400, detail="docs_dir must not be empty")

    docs_path = Path(request.docs_dir)
    if not docs_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"docs_dir is not a valid directory: {request.docs_dir}",
        )

    spec = RAGAppSpec(name="api")
    spec = _apply_overrides(spec, request.config)

    try:
        answer = platform_query_folder(
            docs_dir=request.docs_dir,
            question=request.question,
            spec=spec,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("query-folder failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return _rag_answer_to_response(answer)
