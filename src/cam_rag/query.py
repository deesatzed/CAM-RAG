"""Simple no-LLM document-folder query path."""

from __future__ import annotations

import logging
from pathlib import Path

from cam_rag.documents import read_document_folder
from cam_rag.documents.chunking import chunk_documents
from cam_rag.generation.prompt import build_rag_prompt
from cam_rag.logging_config import StructuredLogger
from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import PipelineExecutor
from cam_rag.pipeline.governance import FitnessTracker, LifecycleManager
from cam_rag.pipeline.registry import TechniqueRegistry, compose_pipeline
from cam_rag.rag.models import (
    Citation,
    CorpusDocument,
    Evidence,
    RAGAnswer,
    RAGTrace,
)
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval import (
    AdaptiveParams,
    DenseVectorRetriever,
    RetrievalDocument,
    SparseBM25Retriever,
    compute_adaptive_params,
    rrf_fuse,
)
from cam_rag.retrieval.query_expansion import build_expanded_query, extract_expansion_terms
from cam_rag.rl.feedback import FeedbackLoop
from cam_rag.verification import (
    score_retrieval_confidence,
    verify_citations_grounded,
)
from cam_rag.verification.sensitive import redact_text, scan_text

logger = logging.getLogger(__name__)
_slog = StructuredLogger(__name__)

_MAX_QUERY_LENGTH = 10_000
_MAX_CORPUS_CHUNKS = 50_000

_SYNTHESIS_KEYWORDS = frozenset(
    {"compare", "contrast", "synthesize", "synthesise", "versus"}
)
_LOGIC_KEYWORDS = frozenset(
    {"why", "how", "because", "reason", "cause", "explain"}
)
_SPEC_PREFIXES = ("what is", "define", "definition of", "describe")

# ---------------------------------------------------------------------------
# Module-level FeedbackLoop cache (one per spec name, lazy-initialized)
# ---------------------------------------------------------------------------

_feedback_loops: dict[str, FeedbackLoop] = {}


def _get_feedback_loop(spec: RAGAppSpec) -> FeedbackLoop:
    """Get or create a FeedbackLoop for the given spec."""
    key = spec.name
    if key not in _feedback_loops:
        state_path = (
            Path(spec.rl_persistence_path)
            if spec.rl_persistence_path is not None
            else None
        )
        _feedback_loops[key] = FeedbackLoop(state_path=state_path)
    return _feedback_loops[key]


# ---------------------------------------------------------------------------
# Module-level governance caches (one per spec name, lazy-initialized)
# ---------------------------------------------------------------------------

_fitness_trackers: dict[str, FitnessTracker] = {}
_lifecycle_managers: dict[str, LifecycleManager] = {}


def _get_governance(
    spec: RAGAppSpec,
) -> tuple[FitnessTracker, LifecycleManager]:
    """Get or create governance objects for the given spec."""
    key = spec.name
    if key not in _fitness_trackers:
        _fitness_trackers[key] = FitnessTracker()
        _lifecycle_managers[key] = LifecycleManager()
    return _fitness_trackers[key], _lifecycle_managers[key]


def classify_query_type(query_text: str) -> str:
    """Classify a query into one of the adaptive-param query types.

    Returns one of: ``"synthesis"``, ``"specification"``, ``"logic"``,
    or ``"summary"`` (the fallback).
    """
    lower = query_text.lower().strip()

    if lower.count("?") > 1 or any(
        kw in lower for kw in _SYNTHESIS_KEYWORDS
    ):
        return "synthesis"

    if any(lower.startswith(prefix) for prefix in _SPEC_PREFIXES):
        return "specification"

    first_word = lower.split()[0] if lower.split() else ""
    if first_word in _LOGIC_KEYWORDS or any(
        kw in lower for kw in _LOGIC_KEYWORDS
    ):
        return "logic"

    return "summary"


def _validate_query(question: str) -> str:
    """Validate and normalize a query string.

    Raises ``ValueError`` for empty, whitespace-only, non-string, or
    over-length queries.
    """
    if not isinstance(question, str):
        raise ValueError("query must be a string")
    cleaned = question.strip()
    if not cleaned:
        raise ValueError(
            "query must not be empty or whitespace-only"
        )
    if len(cleaned) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"query exceeds maximum length of "
            f"{_MAX_QUERY_LENGTH} characters"
        )
    return cleaned


def query(
    question: str,
    documents: list[CorpusDocument],
    spec: RAGAppSpec,
) -> RAGAnswer:
    """Query already-loaded documents.

    This is the app-facing platform API used by Ragamuffin. It mirrors
    `query_document_folder` after ingestion has already happened.
    """
    question = _validate_query(question)
    query_type = classify_query_type(question)
    truncated = question[:120] + "..." if len(question) > 120 else question
    _slog.info(
        "query received",
        query_type=query_type,
        query=truncated,
    )

    chunks = [
        chunk
        for chunk in chunk_documents(
            documents,
            overlap_chars=spec.chunk_overlap,
            strategy=spec.chunk_strategy,
        )
        if spec.accepts_chunk(chunk)
    ]
    evidence, expanded_query, arm_name, generated = _rank_chunks(
        question, chunks, spec, limit=spec.retrieval_top_k
    )

    _slog.info(
        "retrieval complete",
        evidence_count=len(evidence),
    )
    if not evidence:
        _slog.warning("no evidence found", query=truncated)

    trace = RAGTrace(query_type="retrieval_only")
    trace.add("chunk_documents")
    if expanded_query != question:
        trace.add("query_expansion")
    trace.add("hybrid_rank")
    trace.retrieval_stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "evidence": len(evidence),
        "expanded_query": expanded_query,
    }

    answer = _answer_from_evidence(
        question, evidence, trace, spec, pipeline_answer=generated,
    )

    if spec.use_rl and arm_name is not None:
        feedback_loop = _get_feedback_loop(spec)
        feedback_loop.after_query(arm_name, answer)
        trace.add("rl_feedback")
        _slog.info(
            "rl feedback recorded",
            arm_name=arm_name,
        )

    return answer


def query_document_folder(
    docs_dir: str | Path,
    question: str,
    spec: RAGAppSpec,
    *,
    limit: int | None = None,
) -> RAGAnswer:
    """Load a document folder, retrieve evidence, return a cited answer.

    When ``spec.generation_backend`` is set, synthesizes a grounded LLM
    answer from the retrieved evidence. Otherwise returns a retrieval-only
    template answer.
    """

    question = _validate_query(question)
    query_type = classify_query_type(question)
    truncated = question[:120] + "..." if len(question) > 120 else question
    _slog.info(
        "query received",
        query_type=query_type,
        query=truncated,
    )

    documents = read_document_folder(docs_dir, spec)
    chunks = [
        chunk
        for chunk in chunk_documents(
            documents,
            overlap_chars=spec.chunk_overlap,
            strategy=spec.chunk_strategy,
        )
        if spec.accepts_chunk(chunk)
    ]
    top_k = limit or spec.retrieval_top_k
    evidence, expanded_query, arm_name, generated = _rank_chunks(
        question, chunks, spec, limit=top_k
    )

    _slog.info(
        "retrieval complete",
        evidence_count=len(evidence),
    )
    if not evidence:
        _slog.warning("no evidence found", query=truncated)

    trace = RAGTrace(query_type="retrieval_only")
    trace.add("load_documents")
    trace.add("chunk_documents")
    if expanded_query != question:
        trace.add("query_expansion")
    trace.add("hybrid_rank")
    trace.retrieval_stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "evidence": len(evidence),
        "expanded_query": expanded_query,
    }

    answer = _answer_from_evidence(
        question, evidence, trace, spec, pipeline_answer=generated,
    )

    if spec.use_rl and arm_name is not None:
        feedback_loop = _get_feedback_loop(spec)
        feedback_loop.after_query(arm_name, answer)
        trace.add("rl_feedback")
        _slog.info(
            "rl feedback recorded",
            arm_name=arm_name,
        )

    return answer


def _answer_from_evidence(
    query_text: str,
    evidence: list[Evidence],
    trace: RAGTrace,
    spec: RAGAppSpec,
    *,
    pipeline_answer: str = "",
) -> RAGAnswer:

    citations = [
        Citation(
            source=item.chunk.source,
            document_id=item.chunk.document_id,
            title=item.chunk.title,
            section_heading=item.chunk.section_heading,
            excerpt=item.chunk.text[:240],
            score=item.score,
        )
        for item in evidence
    ]
    confidence_report = score_retrieval_confidence(evidence)
    grounding_report = verify_citations_grounded(citations, evidence)
    trace.add("score_confidence")
    trace.add("verify_grounding")
    trace.confidence_details = {
        **confidence_report.to_dict(),
        "grounding": grounding_report.to_dict(),
    }

    _slog.info(
        "confidence scored",
        confidence=confidence_report.overall,
    )
    if confidence_report.overall < 0.4:
        _slog.warning(
            "low confidence",
            confidence=confidence_report.overall,
        )

    # Use pipeline-generated answer if available, otherwise generate here
    if pipeline_answer:
        answer = pipeline_answer
        trace.add("generation")
    else:
        answer = _generate_answer(query_text, evidence, spec)
        if spec.generation_backend is not None and answer != _retrieval_only_answer(query_text, evidence):
            trace.add("generation")

    # --- PHI / PII enforcement ------------------------------------
    redacted_count = 0
    check_pii = spec.policy.enforce_pii
    check_phi = spec.policy.enforce_phi
    if check_pii or check_phi:
        matches = scan_text(
            answer,
            check_pii=check_pii,
            check_phi=check_phi,
            use_llm=spec.policy.use_llm_redaction,
        )
        if matches:
            answer = redact_text(answer, matches)
            redacted_count = len(matches)
            trace.add("sensitive_redaction")
    trace.confidence_details["redacted_count"] = redacted_count

    return RAGAnswer(
        answer=answer,
        evidence=evidence,
        citations=citations,
        confidence=confidence_report.overall,
        grounded=grounding_report.grounded,
        trace=trace,
    )


def _rank_chunks(
    query_text: str,
    chunks,
    spec: RAGAppSpec,
    *,
    limit: int,
) -> tuple[list[Evidence], str, str | None, str]:
    """Rank chunks and optionally generate an answer.

    Returns ``(evidence, expanded_query, arm_name, generated_answer)``.
    ``generated_answer`` is empty when generation is disabled or not
    using the pipeline path.
    """
    # --- P7-04: corpus guard -- truncate oversized chunk lists ---
    if len(chunks) > _MAX_CORPUS_CHUNKS:
        _slog.warning(
            "corpus exceeds maximum chunk limit, truncating",
            chunk_count=len(chunks),
            max_chunks=_MAX_CORPUS_CHUNKS,
        )
        chunks = chunks[:_MAX_CORPUS_CHUNKS]

    # --- M8-05: compute adaptive retrieval parameters ---
    query_type = classify_query_type(query_text)
    adaptive = compute_adaptive_params(
        query_type,
        query_text,
        corpus_size=len(chunks),
    )

    arm_name: str | None = None

    # --- RL-driven pipeline path (opt-in via spec.use_rl) ---
    if spec.use_rl:
        feedback_loop = _get_feedback_loop(spec)
        arm_name, technique_set = feedback_loop.before_query(
            query_type, corpus_size=len(chunks),
        )
        _slog.info(
            "rl arm selected",
            arm_name=arm_name,
            techniques=sorted(technique_set),
        )
        evidence, expanded, generated = _rank_chunks_pipeline(
            query_text,
            chunks,
            spec,
            adaptive=adaptive,
            limit=limit,
            enabled_overrides=technique_set,
        )
        return evidence, expanded, arm_name, generated

    # --- Dynamic pipeline path (opt-in via spec.use_pipeline) ---
    if spec.use_pipeline:
        evidence, expanded, generated = _rank_chunks_pipeline(
            query_text,
            chunks,
            spec,
            adaptive=adaptive,
            limit=limit,
        )
        return evidence, expanded, None, generated

    # --- Static path (default, backward compatibility) ---
    evidence, expanded = _rank_chunks_static(
        query_text,
        chunks,
        spec,
        adaptive=adaptive,
        limit=limit,
    )
    return evidence, expanded, None, ""


def _rank_chunks_pipeline(
    query_text: str,
    chunks,
    spec: RAGAppSpec,
    *,
    adaptive: AdaptiveParams,
    limit: int,
    enabled_overrides: set[str] | None = None,
) -> tuple[list[Evidence], str, str]:
    """Dynamic pipeline path: compose + execute technique sequence.

    Returns ``(evidence, expanded_query, generated_answer)``.
    ``generated_answer`` is empty when no generation backend is configured.
    """
    query_type = classify_query_type(query_text)
    registry = _build_default_registry()
    plan = compose_pipeline(
        registry, query_type, corpus_size=len(chunks),
        enabled_overrides=enabled_overrides,
    )
    ctx = build_context(query_text, chunks, spec, adaptive, limit)

    fitness_tracker = None
    lifecycle_manager = None
    if spec.use_governance:
        fitness_tracker, lifecycle_manager = _get_governance(spec)

    executor = PipelineExecutor()
    executor.run(
        plan, ctx,
        fitness_tracker=fitness_tracker,
        lifecycle_manager=lifecycle_manager,
    )
    return ctx.evidence, ctx.expanded_query, ctx.generated_answer


def _build_default_registry() -> TechniqueRegistry:
    """Build the default technique registry with built-in techniques."""
    from cam_rag.pipeline.registry import TechniqueDescriptor

    registry = TechniqueRegistry()
    techniques = [
        TechniqueDescriptor(
            name="sparse_bm25", category="retrieval"
        ),
        TechniqueDescriptor(
            name="dense_vector", category="retrieval"
        ),
        TechniqueDescriptor(
            name="rrf_fusion",
            category="retrieval",
            requires=("sparse_bm25", "dense_vector"),
        ),
        TechniqueDescriptor(
            name="moe_importance_scoring",
            category="scoring",
            requires=("rrf_fusion",),
        ),
        TechniqueDescriptor(
            name="accuracy_contracts",
            category="scoring",
            requires=("moe_importance_scoring",),
        ),
        TechniqueDescriptor(
            name="selective_filter",
            category="scoring",
            requires=("rrf_fusion", "moe_importance_scoring"),
        ),
        TechniqueDescriptor(
            name="complexity_routing",
            category="routing",
            requires=("rrf_fusion",),
        ),
        TechniqueDescriptor(
            name="query_expansion",
            category="expansion",
            requires=("rrf_fusion",),
            query_types=("summary", "synthesis"),
        ),
        TechniqueDescriptor(
            name="cross_encoder_rerank",
            category="reranking",
            requires=("rrf_fusion",),
        ),
        TechniqueDescriptor(
            name="multi_hop_retrieval",
            category="retrieval",
            requires=("rrf_fusion",),
            query_types=("synthesis", "logic"),
        ),
        TechniqueDescriptor(
            name="confidence_scoring",
            category="verification",
        ),
        TechniqueDescriptor(
            name="citation_grounding",
            category="verification",
            requires=("confidence_scoring",),
        ),
        TechniqueDescriptor(
            name="generation",
            category="generation",
            requires=("citation_grounding",),
        ),
        TechniqueDescriptor(
            name="etf_verification",
            category="verification",
            requires=("generation",),
        ),
    ]
    for t in techniques:
        registry.register(t)
    return registry


def _rank_chunks_static(
    query_text: str,
    chunks,
    spec: RAGAppSpec,
    *,
    adaptive: AdaptiveParams,
    limit: int,
) -> tuple[list[Evidence], str]:
    """Original static path preserved for backward compatibility.

    When ``spec.embedding_backends`` is non-empty, builds one
    ``DenseVectorRetriever`` per backend and fuses sparse + N dense
    result lists via N+1-way RRF.  Otherwise falls back to the
    standard single dense retriever path.
    """
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    documents = [
        RetrievalDocument(
            doc_id=chunk.id,
            text=chunk.text,
            metadata={"chunk_id": chunk.id},
        )
        for chunk in chunks
    ]

    # --- P7-02: build retrievers ONCE and reuse across passes ---
    sparse_retriever = SparseBM25Retriever(
        documents, tokenizer=spec.tokenize
    )

    if spec.embedding_backends:
        # Ensemble mode: build one retriever per backend
        ensemble_retrievers: dict[str, DenseVectorRetriever] = {}
        for i, backend in enumerate(spec.embedding_backends):
            name = f"dense_{i}"
            ensemble_retrievers[name] = DenseVectorRetriever(
                documents, backend=backend,
            )

        evidence = _retrieve_evidence_ensemble(
            query_text,
            sparse_retriever,
            ensemble_retrievers,
            chunk_by_id,
            spec,
            adaptive=adaptive,
            limit=limit,
        )
        expanded_query = query_text
        if spec.query_expansion_enabled and evidence:
            expanded_query = build_expanded_query(
                query_text,
                evidence,
                tokenizer=spec.tokenize,
                max_terms=spec.expansion_terms,
            )
            if expanded_query != query_text:
                evidence = _retrieve_evidence_ensemble(
                    expanded_query,
                    sparse_retriever,
                    ensemble_retrievers,
                    chunk_by_id,
                    spec,
                    adaptive=adaptive,
                    limit=limit,
                )
        # Multi-hop retrieval (after fusion, before reranking)
        evidence = _apply_multi_hop_static(
            query_text,
            evidence,
            sparse_retriever,
            chunk_by_id,
            spec,
            limit=limit,
            adaptive=adaptive,
        )
        evidence = _rerank_evidence(query_text, evidence, spec)
        return evidence, expanded_query

    # Single-backend path (backward compatible)
    dense_retriever = DenseVectorRetriever(
        documents, backend=spec.embedding_backend,
    )

    evidence = _retrieve_evidence(
        query_text,
        sparse_retriever,
        dense_retriever,
        chunk_by_id,
        spec,
        adaptive=adaptive,
        limit=limit,
    )
    expanded_query = query_text
    if spec.query_expansion_enabled and evidence:
        expanded_query = build_expanded_query(
            query_text,
            evidence,
            tokenizer=spec.tokenize,
            max_terms=spec.expansion_terms,
        )
        if expanded_query != query_text:
            # Reuse the same pre-built retrievers for expansion
            evidence = _retrieve_evidence(
                expanded_query,
                sparse_retriever,
                dense_retriever,
                chunk_by_id,
                spec,
                adaptive=adaptive,
                limit=limit,
            )
    # Multi-hop retrieval (after fusion, before reranking)
    evidence = _apply_multi_hop_static(
        query_text,
        evidence,
        sparse_retriever,
        chunk_by_id,
        spec,
        limit=limit,
        adaptive=adaptive,
    )
    evidence = _rerank_evidence(query_text, evidence, spec)
    return evidence, expanded_query


def _apply_multi_hop_static(
    query_text: str,
    evidence: list[Evidence],
    sparse_retriever: SparseBM25Retriever,
    chunk_by_id: dict,
    spec: RAGAppSpec,
    *,
    limit: int,
    adaptive: AdaptiveParams,
) -> list[Evidence]:
    """Run iterative multi-hop retrieval on the static path.

    Mirrors the logic in ``MultiHopRetrievalStep`` but operates on the
    static path's data structures.  Returns the merged evidence list
    unchanged when multi-hop is disabled or no new evidence is found.
    """
    if not spec.multi_hop_enabled:
        return evidence
    if not evidence:
        return evidence

    max_hops = spec.multi_hop_max_hops

    # Tag existing evidence as hop 1
    for item in evidence:
        if "hop" not in item.signals:
            item.signals["hop"] = 1

    existing_chunk_ids = {item.chunk.id for item in evidence}

    for hop_number in range(2, max_hops + 1):
        # Extract new terms from current evidence not in the original query
        new_terms = extract_expansion_terms(
            evidence,
            query_text,
            tokenizer=spec.tokenize,
            top_k=min(5, len(evidence)),
            max_terms=spec.expansion_terms,
        )
        if not new_terms:
            break

        # Build the follow-up query
        follow_up_query = query_text + " " + " ".join(new_terms)

        # Re-run sparse retrieval with the follow-up query
        sparse_k = max(limit, adaptive.sparse_k)
        hop_sparse_results = sparse_retriever.retrieve(
            follow_up_query, k=sparse_k,
        )

        # Convert sparse results to evidence and merge (deduplicate by chunk ID)
        hop_query_terms = set(spec.tokenize(follow_up_query))
        new_evidence: list[Evidence] = []
        for result in hop_sparse_results:
            if result.doc_id in existing_chunk_ids:
                continue
            chunk = chunk_by_id.get(result.doc_id)
            if chunk is None:
                continue
            matched_terms = sorted(
                hop_query_terms.intersection(spec.tokenize(chunk.text))
            )
            new_evidence.append(
                Evidence(
                    chunk=chunk,
                    score=result.score,
                    retriever="multi_hop_sparse",
                    rank=0,
                    signals={
                        "hop": hop_number,
                        "matched_terms": matched_terms,
                        "hop_query": follow_up_query,
                    },
                )
            )
            existing_chunk_ids.add(result.doc_id)

        if not new_evidence:
            break

        # Merge, re-sort, and trim
        evidence.extend(new_evidence)
        evidence.sort(key=lambda e: e.score, reverse=True)
        evidence = evidence[:limit]

        # Update ranks
        for i, item in enumerate(evidence):
            item.rank = i

    return evidence


def _retrieve_evidence(
    query_text: str,
    sparse_retriever: SparseBM25Retriever,
    dense_retriever: DenseVectorRetriever,
    chunk_by_id: dict,
    spec: RAGAppSpec,
    *,
    adaptive: AdaptiveParams,
    limit: int,
) -> list[Evidence]:
    sparse_k = max(limit, adaptive.sparse_k)
    dense_k = max(limit, adaptive.dense_k)

    sparse_results = sparse_retriever.retrieve(
        query_text, k=sparse_k
    )
    dense_results = dense_retriever.retrieve(
        query_text, k=dense_k
    )

    fused_results = rrf_fuse(
        dense_results,
        sparse_results,
        names=["dense", "sparse"],
        rrf_k=adaptive.rrf_k,
        weights={
            "dense": adaptive.dense_weight,
            "sparse": adaptive.sparse_weight,
        },
    )[:limit]

    query_terms = set(spec.tokenize(query_text))
    evidence: list[Evidence] = []
    for result in fused_results:
        chunk = chunk_by_id[result.doc_id]
        matched_terms = sorted(
            query_terms.intersection(spec.tokenize(chunk.text))
        )
        evidence.append(
            Evidence(
                chunk=chunk,
                score=result.rrf_score,
                retriever="hybrid_rrf",
                rank=result.rank,
                signals={
                    "matched_terms": matched_terms,
                    "source_ranks": result.source_ranks,
                    "source_scores": result.source_scores,
                    "rrf_score": result.rrf_score,
                },
            )
        )
    return evidence


def _retrieve_evidence_ensemble(
    query_text: str,
    sparse_retriever: SparseBM25Retriever,
    ensemble_retrievers: dict[str, DenseVectorRetriever],
    chunk_by_id: dict,
    spec: RAGAppSpec,
    *,
    adaptive: AdaptiveParams,
    limit: int,
) -> list[Evidence]:
    """Retrieve evidence with N+1-way RRF fusion (sparse + N dense backends)."""
    sparse_k = max(limit, adaptive.sparse_k)
    dense_k = max(limit, adaptive.dense_k)

    sparse_results = sparse_retriever.retrieve(query_text, k=sparse_k)

    # Build ranked lists: sparse first, then each dense backend
    ranked_lists = [sparse_results]
    names = ["sparse"]
    weights: dict[str, float] = {"sparse": adaptive.sparse_weight}

    num_dense = len(ensemble_retrievers)
    spec_weights = spec.ensemble_weights

    for dense_name, retriever in ensemble_retrievers.items():
        results = retriever.retrieve(query_text, k=dense_k)
        ranked_lists.append(results)
        names.append(dense_name)
        if dense_name in spec_weights:
            weights[dense_name] = spec_weights[dense_name]
        else:
            # Equal share of dense_weight
            weights[dense_name] = adaptive.dense_weight / num_dense

    fused_results = rrf_fuse(
        *ranked_lists,
        names=names,
        rrf_k=adaptive.rrf_k,
        weights=weights,
    )[:limit]

    query_terms = set(spec.tokenize(query_text))
    evidence: list[Evidence] = []
    for result in fused_results:
        chunk = chunk_by_id[result.doc_id]
        matched_terms = sorted(
            query_terms.intersection(spec.tokenize(chunk.text))
        )
        evidence.append(
            Evidence(
                chunk=chunk,
                score=result.rrf_score,
                retriever="hybrid_rrf",
                rank=result.rank,
                signals={
                    "matched_terms": matched_terms,
                    "source_ranks": result.source_ranks,
                    "source_scores": result.source_scores,
                    "rrf_score": result.rrf_score,
                },
            )
        )
    return evidence


def _rerank_evidence(
    query_text: str,
    evidence: list[Evidence],
    spec: RAGAppSpec,
) -> list[Evidence]:
    """Re-score evidence using the configured reranker backend.

    Returns evidence list sorted by reranker score.  When no backend is
    configured, returns the evidence unchanged.
    """
    backend = spec.reranker_backend
    if backend is None:
        return evidence
    if not evidence:
        return evidence

    passages = [item.chunk.text for item in evidence]
    scores = backend.rerank(query_text, passages)

    for item, rerank_score in zip(evidence, scores):
        item.signals["rrf_score_pre_rerank"] = item.score
        item.signals["reranker_score"] = rerank_score
        item.score = rerank_score

    evidence.sort(key=lambda e: e.score, reverse=True)
    for i, item in enumerate(evidence):
        item.rank = i

    _slog.info(
        "reranking complete",
        evidence_count=len(evidence),
        top_score=evidence[0].score if evidence else 0.0,
    )
    return evidence


def _generate_answer(
    query: str,
    evidence: list[Evidence],
    spec: RAGAppSpec,
) -> str:
    """Generate an answer using the LLM backend, or fall back to template."""
    backend = spec.generation_backend
    if backend is None:
        return _retrieval_only_answer(query, evidence)
    if not evidence:
        return f"No cited evidence found for: {query[:200]}"

    system_prompt, user_prompt = build_rag_prompt(
        query, evidence, contract_aware=spec.accuracy_contracts_enabled,
    )
    generated = backend.generate(system_prompt, user_prompt)
    if generated:
        _slog.info("llm generation complete", chars=len(generated))
        return generated
    _slog.warning("generation backend returned empty; using retrieval-only")
    return _retrieval_only_answer(query, evidence)


def _retrieval_only_answer(
    query: str, evidence: list[Evidence]
) -> str:
    display_query = (
        query[:200] + "..." if len(query) > 200 else query
    )
    if not evidence:
        return f"No cited evidence found for: {display_query}"
    lines = [
        f"Retrieved {len(evidence)} cited evidence item(s) "
        f"for: {display_query}"
    ]
    for item in evidence[:3]:
        title = item.chunk.title or item.chunk.source
        lines.append(f"- {title}: {item.chunk.text[:180]}")
    return "\n".join(lines)
