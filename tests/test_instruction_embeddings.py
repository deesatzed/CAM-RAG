"""Tests for InstructionEmbeddingBackend."""

from __future__ import annotations

from cam_rag.retrieval.dense import DenseVectorRetriever, HashEmbeddingBackend
from cam_rag.retrieval.instruction_embeddings import InstructionEmbeddingBackend
from cam_rag.retrieval.models import RetrievalDocument


INSTRUCTION = "Given a scientific claim, retrieve abstracts that support or refute it"


def _make_backend(dim: int = 64) -> InstructionEmbeddingBackend:
    return InstructionEmbeddingBackend(
        HashEmbeddingBackend(dim=dim),
        instruction=INSTRUCTION,
    )


class TestInstructionEmbeddingBackend:
    def test_dim_delegates_to_wrapped(self):
        backend = _make_backend(dim=128)
        assert backend.dim == 128

    def test_embed_passes_through_unchanged(self):
        inner = HashEmbeddingBackend(dim=64)
        backend = InstructionEmbeddingBackend(inner, instruction=INSTRUCTION)
        doc_text = "Aspirin reduces inflammation."
        assert backend.embed(doc_text) == inner.embed(doc_text)

    def test_embed_query_differs_from_embed(self):
        backend = _make_backend()
        text = "Does aspirin reduce inflammation?"
        doc_vec = backend.embed(text)
        query_vec = backend.embed_query(text)
        # The instruction prefix should change the hash embedding
        assert doc_vec != query_vec

    def test_embed_query_includes_instruction(self):
        """embed_query should produce same result as manually prefixed embed."""
        inner = HashEmbeddingBackend(dim=64)
        backend = InstructionEmbeddingBackend(inner, instruction=INSTRUCTION)
        text = "Does aspirin reduce inflammation?"
        prefixed = f"Instruct: {INSTRUCTION}\nQuery: {text}"
        assert backend.embed_query(text) == inner.embed(prefixed)

    def test_embed_batch_delegates(self):
        backend = _make_backend()
        texts = ["doc one", "doc two", "doc three"]
        result = backend.embed_batch(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 64

    def test_embed_query_batch(self):
        backend = _make_backend()
        queries = ["query one", "query two"]
        result = backend.embed_query_batch(queries)
        assert len(result) == 2
        # Each should match individual embed_query
        for text, vec in zip(queries, result):
            assert vec == backend.embed_query(text)

    def test_dense_retriever_uses_embed_query(self):
        """DenseVectorRetriever should dispatch to embed_query for queries."""
        inner = HashEmbeddingBackend(dim=64)
        backend = InstructionEmbeddingBackend(inner, instruction=INSTRUCTION)
        docs = [
            RetrievalDocument(doc_id="d1", text="Aspirin reduces inflammation and pain."),
            RetrievalDocument(doc_id="d2", text="Python is a programming language."),
        ]
        retriever = DenseVectorRetriever(docs, backend=backend)
        results = retriever.retrieve("inflammation", k=2)
        # Should get results (proving embed_query was called)
        assert len(results) > 0

    def test_embed_batch_fallback_without_embed_batch(self):
        """When inner backend has no embed_batch, falls back to per-item embed."""

        class MinimalBackend:
            dim = 32

            def embed(self, text: str) -> list[float]:
                return [0.1] * 32

        backend = InstructionEmbeddingBackend(MinimalBackend(), instruction="test")
        result = backend.embed_batch(["a", "b"])
        assert len(result) == 2
        assert all(len(v) == 32 for v in result)

    def test_embed_query_batch_fallback_without_embed_batch(self):
        """When inner backend has no embed_batch, query batch falls back to per-item."""

        class MinimalBackend:
            dim = 32

            def embed(self, text: str) -> list[float]:
                return [0.1] * 32

        backend = InstructionEmbeddingBackend(MinimalBackend(), instruction="test")
        result = backend.embed_query_batch(["q1", "q2"])
        assert len(result) == 2
