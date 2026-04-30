"""Tests for adaptive IDF fusion weights and AdaptiveFusionStep."""

from __future__ import annotations

import math

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import AdaptiveFusionStep, get_step
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_fusion import compute_idf_adaptive_weights
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import HashEmbeddingBackend


class TestComputeIDFAdaptiveWeights:
    def test_empty_query_returns_base_weights(self):
        d, s = compute_idf_adaptive_weights([], {}, 100)
        assert d == 0.6
        assert s == 0.4

    def test_zero_documents_returns_base_weights(self):
        d, s = compute_idf_adaptive_weights(["term"], {"term": 1}, 0)
        assert d == 0.6
        assert s == 0.4

    def test_weights_sum_to_one(self):
        d, s = compute_idf_adaptive_weights(
            ["rare", "common"],
            {"rare": 1, "common": 500},
            1000,
        )
        assert abs(d + s - 1.0) < 1e-9

    def test_rare_terms_boost_sparse(self):
        """Rare terms (high IDF) should increase sparse weight."""
        df_rare = {"thrombocytopenia": 2, "hemostasis": 3}
        df_common = {"the": 900, "is": 850}
        d_rare, s_rare = compute_idf_adaptive_weights(
            ["thrombocytopenia", "hemostasis"], df_rare, 1000,
        )
        d_common, s_common = compute_idf_adaptive_weights(
            ["the", "is"], df_common, 1000,
        )
        # Rare terms should have higher sparse weight
        assert s_rare > s_common

    def test_common_terms_boost_dense(self):
        """Common terms (low IDF) should increase dense weight."""
        df_common = {"the": 900, "is": 850}
        d, s = compute_idf_adaptive_weights(
            ["the", "is"], df_common, 1000,
        )
        assert d > s  # Dense should dominate for common terms

    def test_custom_base_weights(self):
        d, s = compute_idf_adaptive_weights(
            ["term"],
            {"term": 500},
            1000,
            base_dense_weight=0.7,
            base_sparse_weight=0.3,
        )
        assert abs(d + s - 1.0) < 1e-9

    def test_unknown_terms_treated_as_rare(self):
        """Terms not in document_frequencies have df=0 → high IDF."""
        d, s = compute_idf_adaptive_weights(
            ["unknownterm"],
            {},  # not in corpus at all
            1000,
        )
        # Should boost sparse for very rare terms
        assert s > 0.4  # greater than base sparse weight

    def test_weights_stay_in_valid_range(self):
        """Weights should be clamped to [0.1, 0.9]."""
        # Extreme case: single extremely rare term in huge corpus
        d, s = compute_idf_adaptive_weights(
            ["unicorn"],
            {},
            1_000_000,
            alpha=10.0,
        )
        assert 0.1 <= d <= 0.9
        assert 0.1 <= s <= 0.9


class TestAdaptiveFusionStep:
    def _make_ctx(self, adaptive_fusion_enabled=True):
        chunks = [
            Chunk(id=f"c{i}", document_id="d1", text=f"chunk {i} about medicine {i}")
            for i in range(5)
        ]
        spec = RAGAppSpec(
            name="test",
            use_pipeline=True,
            embedding_backend=HashEmbeddingBackend(dim=32),
            adaptive_fusion_enabled=adaptive_fusion_enabled,
        )
        adaptive = AdaptiveParams(dense_weight=0.6, sparse_weight=0.4)
        return build_context(
            query_text="medicine inflammation",
            chunks=chunks,
            spec=spec,
            adaptive=adaptive,
            limit=5,
        )

    def test_step_is_registered(self):
        step = get_step("adaptive_fusion_weights")
        assert step is not None
        assert isinstance(step, AdaptiveFusionStep)

    def test_noop_when_disabled(self):
        ctx = self._make_ctx(adaptive_fusion_enabled=False)
        step = AdaptiveFusionStep()
        step.execute(ctx)
        # Weights should be unchanged
        assert ctx.adaptive.dense_weight == 0.6
        assert ctx.adaptive.sparse_weight == 0.4

    def test_noop_without_sparse_retriever(self):
        ctx = self._make_ctx()
        ctx.sparse_retriever = None
        step = AdaptiveFusionStep()
        original_dw = ctx.adaptive.dense_weight
        step.execute(ctx)
        assert ctx.adaptive.dense_weight == original_dw

    def test_modifies_weights_when_enabled(self):
        from cam_rag.retrieval.sparse import SparseBM25Retriever

        ctx = self._make_ctx()
        ctx.sparse_retriever = SparseBM25Retriever(ctx.retrieval_docs)
        step = AdaptiveFusionStep()
        step.execute(ctx)
        # Weights should still sum to ~1.0
        assert abs(ctx.adaptive.dense_weight + ctx.adaptive.sparse_weight - 1.0) < 1e-9
