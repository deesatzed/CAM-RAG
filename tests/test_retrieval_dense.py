import pytest

from cam_rag.retrieval import (
    DenseVectorRetriever,
    HashEmbeddingBackend,
    RetrievalDocument,
    cosine_similarity,
)


def test_hash_embedding_backend_is_deterministic_and_normalized():
    backend = HashEmbeddingBackend(dim=32)

    first = backend.embed("ICU escalation protocol")
    second = backend.embed("ICU escalation protocol")

    assert first == second
    assert len(first) == 32
    assert cosine_similarity(first, first) == pytest.approx(1.0)


def test_dense_vector_retriever_finds_related_document():
    retriever = DenseVectorRetriever(
        [
            RetrievalDocument("triage", "emergency department triage vital signs"),
            RetrievalDocument("billing", "invoice review cycle"),
        ],
        backend=HashEmbeddingBackend(dim=64),
    )

    results = retriever.retrieve("triage vital signs", k=1)

    assert results[0].doc_id == "triage"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_cosine_similarity_validates_dimensions():
    with pytest.raises(ValueError, match="same length"):
        cosine_similarity([1.0], [1.0, 0.0])
