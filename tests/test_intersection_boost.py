"""Tests for IntersectionBoostStep."""

from __future__ import annotations

import pytest

from cam_rag.pipeline.context import PipelineContext
from cam_rag.pipeline.executor import get_step
from cam_rag.pipeline.intersection_boost import IntersectionBoostStep
from cam_rag.pipeline.strategy import (
    BUILTIN_STRATEGIES,
    _DENSE_ONLY_STEPS,
    _HYBRID_STEPS,
)
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams


def _make_evidence(
    scores: list[float],
    source_ranks_list: list[dict[str, int]],
) -> list[Evidence]:
    items: list[Evidence] = []
    for i, (score, source_ranks) in enumerate(zip(scores, source_ranks_list)):
        chunk = Chunk(id=f"c{i}", document_id=f"d{i}", text=f"text {i}")
        items.append(
            Evidence(
                chunk=chunk,
                score=score,
                retriever="test",
                rank=i,
                signals={"source_ranks": source_ranks},
            )
        )
    return items


def _make_ctx(evidence: list[Evidence]) -> PipelineContext:
    spec = RAGAppSpec(name="test")
    return PipelineContext(
        query_text="test query",
        chunks=[],
        spec=spec,
        adaptive=AdaptiveParams(),
        limit=10,
        evidence=evidence,
        expanded_query="test query",
    )


class TestIntersectionBoost:
    def test_boosts_dual_source_documents(self) -> None:
        """Documents found by both sparse and dense get 1.2x boost."""
        evidence = _make_evidence(
            [1.0],
            [{"sparse": 1, "dense": 2}],
        )
        ctx = _make_ctx(evidence)
        IntersectionBoostStep().execute(ctx)
        assert ctx.evidence[0].score == pytest.approx(1.2)
        assert ctx.evidence[0].signals["intersection_boost"] is True

    def test_no_boost_for_single_source(self) -> None:
        """Documents from only one retriever get no boost."""
        evidence = _make_evidence(
            [1.0, 0.8],
            [{"dense": 1}, {"sparse": 2}],
        )
        ctx = _make_ctx(evidence)
        IntersectionBoostStep().execute(ctx)
        assert ctx.evidence[0].score == pytest.approx(1.0)
        assert ctx.evidence[0].signals["intersection_boost"] is False
        assert ctx.evidence[1].score == pytest.approx(0.8)
        assert ctx.evidence[1].signals["intersection_boost"] is False

    def test_records_signal_in_evidence(self) -> None:
        """intersection_boost signal is recorded for every item."""
        evidence = _make_evidence(
            [1.0, 0.5],
            [{"sparse": 1, "dense": 3}, {"dense": 2}],
        )
        ctx = _make_ctx(evidence)
        IntersectionBoostStep().execute(ctx)
        assert ctx.evidence[0].signals["intersection_boost"] is True
        assert ctx.evidence[1].signals["intersection_boost"] is False

    def test_noop_on_empty_evidence(self) -> None:
        """No-op when evidence list is empty."""
        ctx = _make_ctx([])
        IntersectionBoostStep().execute(ctx)
        assert ctx.evidence == []

    def test_reorders_after_boost(self) -> None:
        """Boosted item should float to the top after re-sort."""
        evidence = _make_evidence(
            [1.0, 0.9],
            [{"dense": 1}, {"sparse": 1, "dense": 2}],
        )
        ctx = _make_ctx(evidence)
        IntersectionBoostStep().execute(ctx)
        # Item 1 (0.9 * 1.2 = 1.08) should now outrank item 0 (1.0)
        assert ctx.evidence[0].chunk.id == "c1"
        assert ctx.evidence[0].score == pytest.approx(1.08)
        assert ctx.evidence[0].rank == 0
        assert ctx.evidence[1].chunk.id == "c0"
        assert ctx.evidence[1].rank == 1

    def test_handles_ensemble_keys(self) -> None:
        """Handles 'dense_0', 'dense_1', and 'plugin_splade' keys."""
        evidence = _make_evidence(
            [1.0, 0.8, 0.6],
            [
                {"sparse": 1, "dense_0": 2},  # intersection
                {"sparse": 1, "plugin_splade": 3},  # intersection (plugin)
                {"dense_1": 1},  # no sparse → no boost
            ],
        )
        ctx = _make_ctx(evidence)
        IntersectionBoostStep().execute(ctx)
        assert ctx.evidence[0].signals["intersection_boost"] is True
        assert ctx.evidence[1].signals["intersection_boost"] is True
        assert ctx.evidence[2].signals["intersection_boost"] is False

    def test_present_in_hybrid_steps(self) -> None:
        """intersection_boost is in _HYBRID_STEPS."""
        assert "intersection_boost" in _HYBRID_STEPS

    def test_not_in_dense_only_steps(self) -> None:
        """intersection_boost is NOT in _DENSE_ONLY_STEPS."""
        assert "intersection_boost" not in _DENSE_ONLY_STEPS

    def test_step_is_registered(self) -> None:
        """IntersectionBoostStep is discoverable via get_step()."""
        step = get_step("intersection_boost")
        assert step.name == "intersection_boost"
