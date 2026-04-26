"""Simple no-LLM document-folder query path."""

from __future__ import annotations

from pathlib import Path

from cam_rag.documents import read_document_folder
from cam_rag.documents.chunking import chunk_documents
from cam_rag.rag.models import Citation, CorpusDocument, Evidence, RAGAnswer, RAGTrace
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval import RetrievalDocument, SparseBM25Retriever


def query(question: str, documents: list[CorpusDocument], spec: RAGAppSpec) -> RAGAnswer:
    """Query already-loaded documents.

    This is the app-facing platform API used by Ragamuffin. It mirrors
    `query_document_folder` after ingestion has already happened.
    """

    chunks = [chunk for chunk in chunk_documents(documents) if spec.accepts_chunk(chunk)]
    evidence = _rank_chunks(question, chunks, spec, limit=spec.retrieval_top_k)

    trace = RAGTrace(query_type="retrieval_only")
    trace.add("chunk_documents")
    trace.add("lexical_rank")
    trace.retrieval_stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "evidence": len(evidence),
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
    evidence = _rank_chunks(question, chunks, spec, limit=top_k)

    trace = RAGTrace(query_type="retrieval_only")
    trace.add("load_documents")
    trace.add("chunk_documents")
    trace.add("lexical_rank")
    trace.retrieval_stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "evidence": len(evidence),
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
    confidence = evidence[0].score if evidence else 0.0
    answer = _retrieval_only_answer(query_text, evidence)
    return RAGAnswer(
        answer=answer,
        evidence=evidence,
        citations=citations,
        confidence=confidence,
        grounded=bool(evidence),
        trace=trace,
    )


def _rank_chunks(query: str, chunks, spec: RAGAppSpec, *, limit: int) -> list[Evidence]:
    chunk_by_id = {chunk.id: chunk for chunk in chunks}
    documents = [
        RetrievalDocument(
            doc_id=chunk.id,
            text=chunk.text,
            metadata={"chunk_id": chunk.id},
        )
        for chunk in chunks
    ]
    retriever = SparseBM25Retriever(documents, tokenizer=spec.tokenize)
    results = retriever.retrieve(query, k=limit)

    query_terms = set(spec.tokenize(query))
    evidence: list[Evidence] = []
    for result in results:
        chunk = chunk_by_id[result.doc_id]
        matched_terms = sorted(query_terms.intersection(spec.tokenize(chunk.text)))
        evidence.append(
            Evidence(
                chunk=chunk,
                score=result.score,
                retriever="sparse_bm25",
                rank=result.rank,
                signals={
                    "matched_terms": matched_terms,
                    "bm25_score": result.score,
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
