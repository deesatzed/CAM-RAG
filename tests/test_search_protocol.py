"""Tests for CamRAGSearchModel (MTEB SearchProtocol implementation)."""

from __future__ import annotations

from cam_rag.benchmarks.search_protocol import CamRAGSearchModel
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.dense import HashEmbeddingBackend


def _make_model(**spec_kwargs):
    """Build a SearchModel with hash backend for fast tests."""
    defaults = {
        "name": "test-bench",
        "use_pipeline": True,
        "embedding_backend": HashEmbeddingBackend(dim=64),
    }
    defaults.update(spec_kwargs)
    spec = RAGAppSpec(**defaults)
    return CamRAGSearchModel(spec, name="test")


def _fake_corpus():
    """Return a list-of-dicts that looks like an HF Dataset."""
    return [
        {"id": "d1", "title": "Aspirin", "text": "Aspirin reduces inflammation and pain."},
        {"id": "d2", "title": "Ibuprofen", "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug."},
        {"id": "d3", "title": "Acetaminophen", "text": "Acetaminophen relieves pain but is not anti-inflammatory."},
        {"id": "d4", "title": "Naproxen", "text": "Naproxen is used for arthritis and joint pain."},
        {"id": "d5", "title": "Celecoxib", "text": "Celecoxib is a COX-2 selective inhibitor for pain relief."},
    ]


def _fake_queries():
    """Return a list-of-dicts that looks like an HF Dataset of queries."""
    return [
        {"id": "q1", "text": "What drugs reduce inflammation?"},
        {"id": "q2", "text": "Which medication helps with arthritis pain?"},
    ]


class TestSearchModelInstantiation:
    def test_creates_with_default_spec(self):
        model = CamRAGSearchModel()
        assert model.spec is not None
        assert model.mteb_model_meta["name"] == "cam-rag-pipeline"

    def test_creates_with_custom_spec(self):
        model = _make_model()
        assert model.spec.name == "test-bench"
        assert model.mteb_model_meta["name"] == "test"

    def test_has_mteb_model_meta(self):
        model = _make_model()
        meta = model.mteb_model_meta
        assert "name" in meta
        assert "revision" in meta
        assert "release_date" in meta


class TestIndexing:
    def test_index_builds_chunks(self):
        model = _make_model()
        corpus = _fake_corpus()
        model.index(corpus)
        assert len(model._chunks) == 5

    def test_index_builds_chunk_by_id(self):
        model = _make_model()
        model.index(_fake_corpus())
        assert "d1" in model._chunk_by_id
        assert "d5" in model._chunk_by_id

    def test_index_builds_sparse_retriever(self):
        model = _make_model()
        model.index(_fake_corpus())
        assert model._sparse_retriever is not None

    def test_index_builds_dense_retriever(self):
        model = _make_model()
        model.index(_fake_corpus())
        assert model._dense_retriever is not None

    def test_index_handles_title_concatenation(self):
        model = _make_model()
        model.index(_fake_corpus())
        chunk = model._chunk_by_id["d1"]
        assert "Aspirin" in chunk.text
        assert "inflammation" in chunk.text

    def test_index_handles_missing_title(self):
        model = _make_model()
        corpus = [{"id": "x1", "text": "No title here."}]
        model.index(corpus)
        assert model._chunk_by_id["x1"].text == "No title here."


class TestSearch:
    def test_search_returns_dict_of_dicts(self):
        model = _make_model()
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert isinstance(results, dict)
        assert "q1" in results
        assert "q2" in results
        for qid, doc_scores in results.items():
            assert isinstance(doc_scores, dict)
            for doc_id, score in doc_scores.items():
                assert isinstance(doc_id, str)
                assert isinstance(score, (int, float))

    def test_search_returns_scores_greater_than_zero(self):
        model = _make_model()
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        # At least some scores should be > 0 (BM25 + hash should find something)
        all_scores = []
        for doc_scores in results.values():
            all_scores.extend(doc_scores.values())
        assert any(s > 0 for s in all_scores), f"Expected some positive scores, got {all_scores}"

    def test_search_respects_top_k(self):
        model = _make_model()
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=2)
        for doc_scores in results.values():
            assert len(doc_scores) <= 2

    def test_search_empty_corpus_returns_empty_scores(self):
        model = _make_model()
        model.index([])
        results = model.search(_fake_queries(), top_k=3)
        for doc_scores in results.values():
            assert len(doc_scores) == 0

    def test_full_round_trip_bm25_and_dense(self):
        """Full round-trip: index corpus, search, get results with BM25 + dense + RRF."""
        model = _make_model(
            query_expansion_enabled=True,
            expansion_terms=3,
        )
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=5)
        assert len(results) == 2
        # Inflammation query should find aspirin and/or ibuprofen
        q1_scores = results["q1"]
        assert len(q1_scores) > 0

    def test_search_with_moe_scoring(self):
        model = _make_model(moe_scoring_enabled=True)
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2

    def test_search_with_complexity_routing(self):
        model = _make_model(complexity_routing_enabled=True)
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2

    def test_search_with_cross_encoder_reranker(self):
        """Full pipeline with a real local cross-encoder reranker."""
        from cam_rag.reranking.cross_encoder import LocalCrossEncoderBackend

        reranker = LocalCrossEncoderBackend(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu",
        )
        model = _make_model(reranker_backend=reranker)
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2
        # Cross-encoder should re-score evidence — check we still get results
        for doc_scores in results.values():
            assert len(doc_scores) > 0
            for score in doc_scores.values():
                assert isinstance(score, float)

    def test_search_with_adaptive_fusion(self):
        model = _make_model(adaptive_fusion_enabled=True)
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2

    def test_search_with_instruction_prefix(self):
        from cam_rag.retrieval.instruction_embeddings import InstructionEmbeddingBackend

        inner = HashEmbeddingBackend(dim=64)
        instruction_backend = InstructionEmbeddingBackend(
            inner,
            instruction="Given a scientific claim, retrieve supporting abstracts",
        )
        spec = RAGAppSpec(
            name="test-bench",
            use_pipeline=True,
            embedding_backend=instruction_backend,
        )
        model = CamRAGSearchModel(spec, name="test-instruction")
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2
        for doc_scores in results.values():
            assert len(doc_scores) > 0

    def test_search_with_all_strategies_combined(self):
        """Full pipeline: cross-encoder + adaptive fusion + instruction prefix."""
        from cam_rag.reranking.cross_encoder import LocalCrossEncoderBackend
        from cam_rag.retrieval.instruction_embeddings import InstructionEmbeddingBackend

        inner = HashEmbeddingBackend(dim=64)
        instruction_backend = InstructionEmbeddingBackend(
            inner,
            instruction="Given a scientific claim, retrieve supporting abstracts",
        )
        reranker = LocalCrossEncoderBackend(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu",
        )
        spec = RAGAppSpec(
            name="test-all-strategies",
            use_pipeline=True,
            embedding_backend=instruction_backend,
            reranker_backend=reranker,
            adaptive_fusion_enabled=True,
            query_expansion_enabled=True,
            expansion_terms=3,
        )
        model = CamRAGSearchModel(spec, name="test-combined")
        model.index(_fake_corpus())
        results = model.search(_fake_queries(), top_k=3)
        assert len(results) == 2
        for doc_scores in results.values():
            assert len(doc_scores) > 0


class TestSearchProtocolInterface:
    def test_has_index_method(self):
        model = _make_model()
        assert callable(getattr(model, "index", None))

    def test_has_search_method(self):
        model = _make_model()
        assert callable(getattr(model, "search", None))

    def test_has_mteb_model_meta_attribute(self):
        model = _make_model()
        assert hasattr(model, "mteb_model_meta")
        assert isinstance(model.mteb_model_meta, dict)
