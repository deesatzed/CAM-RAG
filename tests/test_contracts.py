from cam_rag import Chunk, CorpusDocument, Evidence, RAGAnswer, RAGAppSpec
from cam_rag.rag import RAGPolicy


def test_generic_app_spec_defaults():
    spec = RAGAppSpec(name="generic")

    assert spec.tokenize("Hybrid RAG") == ["hybrid", "rag"]
    assert ".pdf" in spec.supported_extensions
    assert ".docx" in spec.supported_extensions
    assert spec.policy.require_citations is True
    assert spec.query_expansion_enabled is False
    assert spec.expansion_terms == 5


def test_ragamuffin_can_be_represented_as_app_spec():
    spec = RAGAppSpec(
        name="ragamuffin",
        description="Clinical document-folder RAG",
        policy=RAGPolicy(enforce_phi=True, enforce_pii=True, allowed_residency="US"),
        domain_tags=("clinical", "protocols"),
    )

    assert spec.policy.enforce_phi is True
    assert spec.policy.enforce_pii is True
    assert spec.domain_tags == ("clinical", "protocols")


def test_answer_keeps_graph_reasons_for_repofrax_port():
    document = CorpusDocument(
        id="doc-1",
        text="Use reciprocal rank fusion to combine dense and sparse retrieval.",
        source="note.md",
    )
    chunk = Chunk(id="chunk-1", document_id=document.id, text=document.text)
    evidence = Evidence(
        chunk=chunk,
        score=0.9,
        retriever="methodology_graph",
        graph_reasons=["from:hybrid-retrieval", "shared_category"],
    )
    answer = RAGAnswer(answer="Use RRF.", evidence=[evidence], confidence=0.9)

    assert answer.evidence[0].graph_reasons == ["from:hybrid-retrieval", "shared_category"]
    assert answer.confidence == 0.9
