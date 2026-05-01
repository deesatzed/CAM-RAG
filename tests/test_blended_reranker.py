"""Tests for blended reranker scoring in CrossEncoderRerankStep."""

from __future__ import annotations

import pytest

from cam_rag.pipeline.context import PipelineContext
from cam_rag.pipeline.executor import CrossEncoderRerankStep
from cam_rag.pipeline.strategy import PipelineStrategy
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams


class _FakeReranker:
    """Deterministic reranker that returns fixed scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return self._scores[: len(passages)]


def _make_evidence(scores: list[float]) -> list[Evidence]:
    items: list[Evidence] = []
    for i, score in enumerate(scores):
        chunk = Chunk(id=f"c{i}", document_id=f"d{i}", text=f"text {i}")
        items.append(
            Evidence(
                chunk=chunk,
                score=score,
                retriever="test",
                rank=i,
                signals={},
            )
        )
    return items


def _make_ctx(
    evidence: list[Evidence],
    reranker_scores: list[float],
    *,
    blend_weight_spec: float = 1.0,
    blend_weight_extras: float | None = None,
) -> PipelineContext:
    spec = RAGAppSpec(
        name="test",
        reranker_backend=_FakeReranker(reranker_scores),
        rerank_blend_weight=blend_weight_spec,
    )
    ctx = PipelineContext(
        query_text="test query",
        chunks=[],
        spec=spec,
        adaptive=AdaptiveParams(),
        limit=10,
        evidence=evidence,
        expanded_query="test query",
    )
    if blend_weight_extras is not None:
        ctx.extras["rerank_blend_weight"] = blend_weight_extras
    return ctx


class TestBlendedReranker:
    def test_blend_weight_1_replaces_score(self) -> None:
        """blend_weight=1.0 (default) fully replaces score — current behavior."""
        evidence = _make_evidence([0.5, 0.3])
        ctx = _make_ctx(evidence, [0.9, 0.8], blend_weight_spec=1.0)
        CrossEncoderRerankStep().execute(ctx)
        assert ctx.evidence[0].score == pytest.approx(0.9)
        assert ctx.evidence[1].score == pytest.approx(0.8)

    def test_blend_weight_0_keeps_original_score(self) -> None:
        """blend_weight=0.0 keeps the original score entirely."""
        evidence = _make_evidence([0.5, 0.3])
        ctx = _make_ctx(evidence, [0.9, 0.8], blend_weight_spec=0.0)
        CrossEncoderRerankStep().execute(ctx)
        assert ctx.evidence[0].score == pytest.approx(0.5)
        assert ctx.evidence[1].score == pytest.approx(0.3)

    def test_blend_weight_0_2_blends_80_20(self) -> None:
        """blend_weight=0.2 → 80% original + 20% reranker."""
        evidence = _make_evidence([0.5, 0.3])
        ctx = _make_ctx(evidence, [0.9, 0.1], blend_weight_spec=0.2)
        CrossEncoderRerankStep().execute(ctx)
        # item 0: 0.8 * 0.5 + 0.2 * 0.9 = 0.40 + 0.18 = 0.58
        assert ctx.evidence[0].score == pytest.approx(0.58)
        # item 1: 0.8 * 0.3 + 0.2 * 0.1 = 0.24 + 0.02 = 0.26
        assert ctx.evidence[1].score == pytest.approx(0.26)

    def test_blend_weight_from_spec_fallback(self) -> None:
        """When extras has no blend weight, reads from spec."""
        evidence = _make_evidence([1.0])
        ctx = _make_ctx(evidence, [0.0], blend_weight_spec=0.5)
        CrossEncoderRerankStep().execute(ctx)
        # 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert ctx.evidence[0].score == pytest.approx(0.5)

    def test_extras_overrides_spec(self) -> None:
        """ctx.extras['rerank_blend_weight'] overrides spec."""
        evidence = _make_evidence([1.0])
        ctx = _make_ctx(
            evidence, [0.0],
            blend_weight_spec=1.0,  # spec says full replacement
            blend_weight_extras=0.0,  # extras says keep original
        )
        CrossEncoderRerankStep().execute(ctx)
        # extras wins: blend_weight=0.0 → keeps original score 1.0
        assert ctx.evidence[0].score == pytest.approx(1.0)

    def test_signal_recorded_in_evidence(self) -> None:
        """blend_weight is recorded in evidence signals."""
        evidence = _make_evidence([0.5])
        ctx = _make_ctx(evidence, [0.9], blend_weight_spec=0.2)
        CrossEncoderRerankStep().execute(ctx)
        assert ctx.evidence[0].signals["rerank_blend_weight"] == 0.2
        assert ctx.evidence[0].signals["reranker_score"] == pytest.approx(0.9)
        assert ctx.evidence[0].signals["rrf_score_pre_rerank"] == pytest.approx(0.5)

    def test_pipeline_strategy_carries_field(self) -> None:
        """PipelineStrategy has rerank_blend_weight with default 1.0."""
        strategy = PipelineStrategy(
            name="test",
            steps=("dense_vector",),
            dense_weight=1.0,
            sparse_weight=0.0,
            retrieval_depth=100,
        )
        assert strategy.rerank_blend_weight == 1.0

        custom = PipelineStrategy(
            name="custom",
            steps=("dense_vector",),
            dense_weight=1.0,
            sparse_weight=0.0,
            retrieval_depth=100,
            rerank_blend_weight=0.2,
        )
        assert custom.rerank_blend_weight == 0.2

    def test_noop_when_no_reranker_backend(self) -> None:
        """No-op when reranker_backend is None."""
        evidence = _make_evidence([0.5, 0.3])
        spec = RAGAppSpec(name="test", reranker_backend=None)
        ctx = PipelineContext(
            query_text="test query",
            chunks=[],
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=10,
            evidence=evidence,
            expanded_query="test query",
        )
        CrossEncoderRerankStep().execute(ctx)
        # Scores unchanged
        assert ctx.evidence[0].score == pytest.approx(0.5)
        assert ctx.evidence[1].score == pytest.approx(0.3)
