"""Tests for corpus-aware strategy routing in StrategyRouter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cam_rag.pipeline.strategy import (
    BUILTIN_STRATEGIES,
    STRONG_HYBRID,
    StrategyRouter,
)
from cam_rag.retrieval.corpus_signals import CorpusSignals


def _make_spec(strategy: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(pipeline_strategy=strategy)


def _make_signals(
    *,
    avg_doc_length: float = 150.0,
    doc_length_std: float = 50.0,
    corpus_size: int = 5000,
    vocab_overlap_ratio: float = 0.2,
    unique_term_ratio: float = 0.1,
    short_doc_fraction: float = 0.3,
) -> CorpusSignals:
    return CorpusSignals(
        avg_doc_length=avg_doc_length,
        doc_length_std=doc_length_std,
        corpus_size=corpus_size,
        vocab_overlap_ratio=vocab_overlap_ratio,
        unique_term_ratio=unique_term_ratio,
        short_doc_fraction=short_doc_fraction,
    )


class TestStrongHybridStrategy:
    """Verify the STRONG_HYBRID strategy is registered and configured."""

    def test_strong_hybrid_in_builtins(self) -> None:
        assert "strong_hybrid" in BUILTIN_STRATEGIES

    def test_strong_hybrid_weights(self) -> None:
        assert STRONG_HYBRID.dense_weight == 0.7
        assert STRONG_HYBRID.sparse_weight == 0.3

    def test_strong_hybrid_has_bm25(self) -> None:
        assert "sparse_bm25" in STRONG_HYBRID.steps
        assert "rrf_fusion" in STRONG_HYBRID.steps

    def test_strong_hybrid_no_dense_to_evidence(self) -> None:
        assert "dense_to_evidence" not in STRONG_HYBRID.steps


class TestCorpusAwareRouting:
    """Test that corpus signals override tier-based strategy selection."""

    def test_strong_short_docs_routes_to_strong_hybrid(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(short_doc_fraction=0.85)
        result = router.select("strong", _make_spec(), corpus_signals=signals)
        assert result.name == "strong_hybrid"

    def test_strong_long_docs_routes_to_dense_dominant(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(short_doc_fraction=0.1, vocab_overlap_ratio=0.1)
        result = router.select("strong", _make_spec(), corpus_signals=signals)
        assert result.name == "dense_dominant"

    def test_strong_high_vocab_overlap_routes_to_strong_hybrid(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(vocab_overlap_ratio=0.4, short_doc_fraction=0.2)
        result = router.select("strong", _make_spec(), corpus_signals=signals)
        assert result.name == "strong_hybrid"

    def test_moderate_unaffected_by_corpus_signals(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(short_doc_fraction=0.9, vocab_overlap_ratio=0.5)
        result = router.select("moderate", _make_spec(), corpus_signals=signals)
        assert result.name == "hybrid"

    def test_weak_unaffected_by_short_docs(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(short_doc_fraction=0.9)
        result = router.select("weak", _make_spec(), corpus_signals=signals)
        # weak normally goes to sparse_boost, but corpus_size=5000 so no override
        assert result.name == "sparse_boost"

    def test_weak_small_corpus_routes_to_hybrid(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(corpus_size=500)
        result = router.select("weak", _make_spec(), corpus_signals=signals)
        assert result.name == "hybrid"

    def test_explicit_strategy_overrides_corpus_signals(self) -> None:
        router = StrategyRouter()
        signals = _make_signals(short_doc_fraction=0.9)
        result = router.select(
            "strong", _make_spec("dense_dominant"), corpus_signals=signals,
        )
        assert result.name == "dense_dominant"

    def test_no_corpus_signals_preserves_existing_behavior(self) -> None:
        router = StrategyRouter()
        assert router.select("strong", _make_spec()).name == "dense_dominant"
        assert router.select("moderate", _make_spec()).name == "hybrid"
        assert router.select("weak", _make_spec()).name == "sparse_boost"


class TestNFCorpusScenario:
    """End-to-end test mimicking NFCorpus characteristics."""

    def test_nfcorpus_like_corpus_strong_embeddings(self) -> None:
        """NFCorpus: short biomedical docs, high vocab overlap, strong embeddings."""
        router = StrategyRouter()
        signals = _make_signals(
            avg_doc_length=80.0,
            short_doc_fraction=0.85,
            vocab_overlap_ratio=0.35,
            corpus_size=3633,
        )
        result = router.select("strong", _make_spec(), corpus_signals=signals)
        assert result.name == "strong_hybrid"
        assert result.dense_weight == 0.7
        assert result.sparse_weight == 0.3
        assert "sparse_bm25" in result.steps

    def test_scifact_like_corpus_strong_embeddings(self) -> None:
        """SciFact: longer scientific docs, lower overlap, strong embeddings."""
        router = StrategyRouter()
        signals = _make_signals(
            avg_doc_length=250.0,
            short_doc_fraction=0.1,
            vocab_overlap_ratio=0.15,
            corpus_size=5183,
        )
        result = router.select("strong", _make_spec(), corpus_signals=signals)
        assert result.name == "dense_dominant"
        assert result.dense_weight == 1.0
        assert "sparse_bm25" not in result.steps
