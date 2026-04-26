"""Simple no-LLM document-folder query path."""

from __future__ import annotations

from pathlib import Path

from cam_rag.documents import read_document_folder
from cam_rag.documents.chunking import chunk_documents
from cam_rag.rag.models import Citation, CorpusDocument, Evidence, RAGAnswer, RAGTrace
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval import DenseVectorRetriever, RetrievalDocument, SparseBM25Retriever, rrf_fuse
from cam_rag.retrieval.query_expansion import build_expanded_query
from cam_rag.verification import score_retrieval_confidence, verify_citations_grounded


def query(question: str, documents: list[CorpusDocument], spec: RAGAppSpec) -> RAGAnswer:
    """Query already-loaded documents.

    This is the app-facing platform API used by Ragamuffin. It mirrors
    `query_document_folder` after ingestion has already happened.
    """

    chunks = [chunk for chunk in chunk_documents(documents) if spec.accepts_chunk(chunk)]
    evidence, expanded_query = _rank_chunks(question, chunks, spec, limit=spec.retrieval_top_k)

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
    return _answer_from_evidence(question, evidence, trace)


def query_document_folder(
    docs_dir: str | Path,
    question: str,
    spec: RAGAppSpec,
    *,
    limit: int | None = None,
) -> RAGAnswer:
    """Load a document folder, retrieve evidence, and return a cited answer.

    This intentionally avoids generation. It establishes the app/platform path:
    folder -> documents -> chunks -> ranked evidence -> citations. Dense search,
    RRF, reranking, and synthesis can replace the scorer while preserving the
    public shape.
    """

    documents = read_document_folder(docs_dir, spec)
    chunks = [chunk for chunk in chunk_documents(documents) if spec.accepts_chunk(chunk)]
    top_k = limit or spec.retrieval_top_k
    evidence, expanded_query = _rank_chunks(question, chunks, spec, limit=top_k)

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
    return _answer_from_evidence(question, evidence, trace)


def _answer_from_evidence(query_text: str, evidence: list[Evidence], trace: RAGTrace) -> RAGAnswer:

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
    answer = _retrieval_only_answer(query_text, evidence)
    return RAGAnswer(
        answer=answer,
        evidence=evidence,
        citations=citations,
        confidence=confidence_report.overall,
        grounded=grounding_report.grounded,
        trace=trace,
    )


def _rank_chunks(query: str, chunks, spec: RAGAppSpec, *, limit: int) -> tuple[list[Evidence], str]:
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    documents = [
        RetrievalDocument(
            doc_id=chunk.id,
            text=chunk.text,
            metadata={"chunk_id": chunk.id},
        )
        for chunk in chunks
    ]
    evidence = _retrieve_evidence(query, documents, chunk_by_id, spec, limit=limit)
    expanded_query = query
    if spec.query_expansion_enabled and evidence:
        expanded_query = build_expanded_query(
            query,
            evidence,
            tokenizer=spec.tokenize,
            max_terms=spec.expansion_terms,
        )
        if expanded_query != query:
            evidence = _retrieve_evidence(expanded_query, documents, chunk_by_id, spec, limit=limit)
    return evidence, expanded_query


def _retrieve_evidence(
    query: str,
    documents: list[RetrievalDocument],
    chunk_by_id: dict,
    spec: RAGAppSpec,
    *,
    limit: int,
) -> list[Evidence]:
    sparse_results = SparseBM25Retriever(documents, tokenizer=spec.tokenize).retrieve(
        query, k=max(limit, spec.retrieval_top_k)
    )
    dense_results = DenseVectorRetriever(documents).retrieve(
        query,
        k=max(limit, spec.retrieval_top_k),
    )
    fused_results = rrf_fuse(
        dense_results,
        sparse_results,
        names=["dense", "sparse"],
        weights={"dense": spec.dense_weight, "sparse": spec.sparse_weight},
    )[:limit]

    query_terms = set(spec.tokenize(query))
    evidence: list[Evidence] = []
    for result in fused_results:
        chunk = chunk_by_id[result.doc_id]
        matched_terms = sorted(query_terms.intersection(spec.tokenize(chunk.text)))
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


def _retrieval_only_answer(query: str, evidence: list[Evidence]) -> str:
    if not evidence:
        return f"No cited evidence found for: {query}"
    lines = [f"Retrieved {len(evidence)} cited evidence item(s) for: {query}"]
    for item in evidence[:3]:
        title = item.chunk.title or item.chunk.source
        lines.append(f"- {title}: {item.chunk.text[:180]}")
    return "\n".join(lines)
