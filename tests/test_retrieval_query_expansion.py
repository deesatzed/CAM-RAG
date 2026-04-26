from cam_rag.rag.models import Chunk, Evidence
from cam_rag.retrieval import (
    FusedResult,
    RetrievalResult,
    build_expanded_query,
    extract_expansion_terms,
    filter_expansion_terms,
)


def test_filter_expansion_terms_removes_query_terms_stopwords_and_duplicates():
    terms = ["Graph", "the", "triage", "fusion", "Fusion", "v2", "99"]

    assert filter_expansion_terms(terms, "graph triage") == ["fusion"]


def test_extract_expansion_terms_from_retrieval_and_fused_results():
    items = [
        RetrievalResult(
            "a",
            "Graph retrieval uses reciprocal rank fusion for evidence routing.",
            0.9,
            rank=1,
        ),
        FusedResult(
            doc_id="b",
            rrf_score=0.2,
            rank=2,
            source_ranks={"sparse": 1},
            source_scores={"sparse": 4.0},
            text="Fusion routing keeps retrieval evidence stable.",
        ),
        RetrievalResult("c", "ignored rare expansion term", 0.1, rank=3),
    ]

    terms = extract_expansion_terms(items, "graph retrieval", top_k=2, max_terms=4)

    assert terms == ["fusion", "evidence", "routing", "uses"]


def test_extract_expansion_terms_reads_evidence_like_chunk_text():
    evidence = Evidence(
        chunk=Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="BM25 sparse retrieval complements dense embeddings and chunk reranking.",
        ),
        score=1.0,
        retriever="hybrid",
    )

    terms = extract_expansion_terms([evidence], "dense retrieval", max_terms=3)

    assert terms == ["bm25", "sparse", "complements"]


def test_build_expanded_query_appends_limited_terms_and_preserves_empty_results():
    items = [
        {"chunk": {"text": "incident escalation policy with outage triage runbook"}},
        {"text": "this result should be ignored when top_k is one"},
    ]

    expanded = build_expanded_query(
        "outage triage",
        items,
        top_k=1,
        max_terms=2,
        stopwords={"incident"},
    )

    assert expanded == "outage triage escalation policy"
    assert build_expanded_query("outage triage", [], max_terms=2) == "outage triage"
