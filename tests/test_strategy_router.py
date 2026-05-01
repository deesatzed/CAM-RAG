"""Tests for PipelineStrategy and StrategyRouter."""

from __future__ import annotations

import pytest

from cam_rag.pipeline.strategy import (
    BLENDED_RERANK,
    BUILTIN_STRATEGIES,
    DENSE_DOMINANT,
    HYBRID,
    SPARSE_BOOST,
    PipelineStrategy,
    StrategyRouter,
)
from cam_rag.rag.spec import RAGAppSpec


class TestPipelineStrategy:
    def test_dense_dominant_has_no_sparse_steps(self) -> None:
        assert "sparse_bm25" not in DENSE_DOMINANT.steps
        assert "rrf_fusion" not in DENSE_DOMINANT.steps
        assert "dense_to_evidence" in DENSE_DOMINANT.steps

    def test_dense_dominant_weights(self) -> None:
        assert DENSE_DOMINANT.dense_weight == 1.0
        assert DENSE_DOMINANT.sparse_weight == 0.0

    def test_hybrid_has_both_retrievers(self) -> None:
        assert "sparse_bm25" in HYBRID.steps
        assert "dense_vector" in HYBRID.steps
        assert "rrf_fusion" in HYBRID.steps

    def test_hybrid_default_weights(self) -> None:
        assert HYBRID.dense_weight == 0.6
        assert HYBRID.sparse_weight == 0.4

    def test_sparse_boost_weights(self) -> None:
        assert SPARSE_BOOST.dense_weight == 0.3
        assert SPARSE_BOOST.sparse_weight == 0.7

    def test_sparse_boost_deeper_retrieval(self) -> None:
        assert SPARSE_BOOST.retrieval_depth == 150
        assert HYBRID.retrieval_depth == 100

    def test_most_strategies_have_reranker(self) -> None:
        # dense_only intentionally skips the reranker
        for name, strategy in BUILTIN_STRATEGIES.items():
            if name == "dense_only":
                assert "cross_encoder_rerank" not in strategy.steps
            else:
                assert "cross_encoder_rerank" in strategy.steps

    def test_strategy_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            HYBRID.name = "modified"  # type: ignore[misc]


class TestStrategyRouter:
    def test_select_strong_returns_dense_only(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="auto")
        router = StrategyRouter()
        strategy = router.select("strong", spec)
        assert strategy.name == "dense_only"

    def test_select_moderate_returns_hybrid(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="auto")
        router = StrategyRouter()
        strategy = router.select("moderate", spec)
        assert strategy.name == "hybrid"

    def test_select_weak_returns_sparse_boost(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="auto")
        router = StrategyRouter()
        strategy = router.select("weak", spec)
        assert strategy.name == "sparse_boost"

    def test_explicit_strategy_overrides_tier(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="dense_dominant")
        router = StrategyRouter()
        # Even with "weak" tier, explicit strategy wins
        strategy = router.select("weak", spec)
        assert strategy.name == "dense_dominant"

    def test_explicit_hybrid_overrides_strong_tier(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="hybrid")
        router = StrategyRouter()
        strategy = router.select("strong", spec)
        assert strategy.name == "hybrid"

    def test_unknown_explicit_strategy_raises(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="nonexistent")
        router = StrategyRouter()
        with pytest.raises(KeyError, match="Unknown pipeline strategy"):
            router.select("moderate", spec)

    def test_unknown_tier_defaults_to_hybrid(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="auto")
        router = StrategyRouter()
        strategy = router.select("unknown_tier", spec)
        assert strategy.name == "hybrid"

    def test_register_custom_strategy(self) -> None:
        custom = PipelineStrategy(
            name="custom_fast",
            steps=("dense_vector", "dense_to_evidence"),
            dense_weight=1.0,
            sparse_weight=0.0,
            retrieval_depth=50,
            description="Ultra-fast dense-only",
        )
        router = StrategyRouter()
        router.register_strategy(custom)
        spec = RAGAppSpec(name="test", pipeline_strategy="custom_fast")
        strategy = router.select("moderate", spec)
        assert strategy.name == "custom_fast"
        assert strategy.retrieval_depth == 50

    def test_available_strategies(self) -> None:
        router = StrategyRouter()
        available = router.available
        assert "dense_dominant" in available
        assert "hybrid" in available
        assert "sparse_boost" in available

    def test_custom_strategies_dict_in_constructor(self) -> None:
        custom = PipelineStrategy(
            name="only_strategy",
            steps=("dense_vector",),
            dense_weight=1.0,
            sparse_weight=0.0,
            retrieval_depth=10,
        )
        router = StrategyRouter(strategies={"only_strategy": custom})
        assert router.available == ["only_strategy"]


class TestBlendedRerankStrategy:
    def test_blended_rerank_in_builtin_strategies(self) -> None:
        assert "blended_rerank" in BUILTIN_STRATEGIES

    def test_has_cross_encoder_rerank_in_steps(self) -> None:
        assert "cross_encoder_rerank" in BLENDED_RERANK.steps

    def test_rerank_blend_weight_is_0_2(self) -> None:
        assert BLENDED_RERANK.rerank_blend_weight == 0.2

    def test_explicit_selection_via_router(self) -> None:
        spec = RAGAppSpec(name="test", pipeline_strategy="blended_rerank")
        router = StrategyRouter()
        strategy = router.select("strong", spec)
        assert strategy.name == "blended_rerank"
        assert strategy.rerank_blend_weight == 0.2
