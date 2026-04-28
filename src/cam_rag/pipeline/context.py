"""Pipeline execution context that flows through technique steps.

``PipelineContext`` is the mutable bag of state that technique steps read
from and write to.  It replaces the many positional arguments that the
static ``_rank_chunks`` / ``_retrieve_evidence`` functions pass between
phases, making it straightforward to add new techniques that produce or
consume new intermediate artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cam_rag.rag.models import Chunk, Citation, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import DenseVectorRetriever
from cam_rag.retrieval.models import FusedResult, RetrievalDocument, RetrievalResult
from cam_rag.retrieval.sparse import SparseBM25Retriever


@dataclass(slots=True)
class PipelineContext:
    """Mutable state bag that flows through every technique step.

    Core fields are set at construction time.  Technique steps populate
    the ``intermediate`` fields as they execute.
    """

    # --- Inputs (set before execution) ------------------------------------
    query_text: str
    chunks: list[Chunk]
    spec: RAGAppSpec
    adaptive: AdaptiveParams
    limit: int

    # --- Derived (set during init or early steps) -------------------------
    chunk_by_id: dict[str, Chunk] = field(default_factory=dict)
    retrieval_docs: list[RetrievalDocument] = field(default_factory=list)

    # --- Intermediate results (populated by technique steps) ---------------
    sparse_retriever: SparseBM25Retriever | None = None
    dense_retriever: DenseVectorRetriever | None = None
    sparse_results: list[RetrievalResult] = field(default_factory=list)
    dense_results: list[RetrievalResult] = field(default_factory=list)
    fused_results: list[FusedResult] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    expanded_query: str = ""

    # --- Verification results (populated by confidence/grounding steps) ----
    citations: list[Citation] = field(default_factory=list)
    confidence_score: float | None = None
    grounded: bool | None = None
    confidence_details: dict[str, Any] = field(default_factory=dict)

    # --- Generation result (populated by GenerationStep) -------------------
    generated_answer: str = ""

    # --- Extension slot for custom technique steps -------------------------
    extras: dict[str, Any] = field(default_factory=dict)

    # --- Trace / diagnostics -----------------------------------------------
    steps_executed: list[str] = field(default_factory=list)
    step_timings: dict[str, float] = field(default_factory=dict)


def build_context(
    query_text: str,
    chunks: list[Chunk],
    spec: RAGAppSpec,
    adaptive: AdaptiveParams,
    limit: int,
) -> PipelineContext:
    """Build a fully-initialised context ready for pipeline execution."""
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    retrieval_docs = [
        RetrievalDocument(
            doc_id=chunk.id,
            text=chunk.text,
            metadata={"chunk_id": chunk.id},
        )
        for chunk in chunks
    ]
    return PipelineContext(
        query_text=query_text,
        chunks=chunks,
        spec=spec,
        adaptive=adaptive,
        limit=limit,
        chunk_by_id=chunk_by_id,
        retrieval_docs=retrieval_docs,
        expanded_query=query_text,
    )
