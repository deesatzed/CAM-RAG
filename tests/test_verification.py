from cam_rag.rag.models import Chunk, Citation, Evidence
from cam_rag.verification import score_retrieval_confidence, verify_citations_grounded


def test_score_retrieval_confidence_rewards_agreement_and_coverage():
    chunk = Chunk(id="c1", document_id="d1", text="ED triage vital signs", source="triage.md")
    evidence = [
        Evidence(
            chunk=chunk,
            score=0.02,
            retriever="hybrid_rrf",
            signals={
                "matched_terms": ["ed", "triage", "vital", "signs"],
                "source_ranks": {"dense": 1, "sparse": 1},
                "source_scores": {"dense": 0.8, "sparse": 1.5},
            },
        )
    ]

    report = score_retrieval_confidence(evidence)

    assert report.overall > 0.7
    assert report.source_agreement == 1.0
    assert report.query_coverage == 1.0


def test_score_retrieval_confidence_empty_is_zero():
    assert score_retrieval_confidence([]).overall == 0.0


def test_verify_citations_grounded_matches_evidence_sources():
    chunk = Chunk(id="c1", document_id="d1", text="protocol", source="triage.md")
    evidence = [Evidence(chunk=chunk, score=1.0, retriever="test")]
    citations = [Citation(source="triage.md", document_id="d1")]

    report = verify_citations_grounded(citations, evidence)

    assert report.grounded is True
    assert report.matched_citations == 1


def test_verify_citations_grounded_reports_missing_sources():
    chunk = Chunk(id="c1", document_id="d1", text="protocol", source="triage.md")
    evidence = [Evidence(chunk=chunk, score=1.0, retriever="test")]
    citations = [Citation(source="other.md", document_id="d2")]

    report = verify_citations_grounded(citations, evidence)

    assert report.grounded is False
    assert report.missing_sources == ["other.md"]
