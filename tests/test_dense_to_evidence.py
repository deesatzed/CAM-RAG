"""Tests for DenseToEvidenceStep."""

from __future__ import annotations

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import DenseToEvidenceStep, get_step
from cam_rag.rag.models import Chunk
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.models import RetrievalResult


def _make_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(
            id=f"doc_{i}",
            document_id=f"doc_{i}",
            text=f"This is document {i} about topic {i}",
        )
        for i in range(n)
    ]


def _make_dense_results(chunk_ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            doc_id=doc_id,
            text=f"text for {doc_id}",
            score=1.0 - i * 0.1,
            rank=i + 1,
            metadata={},
        )
        for i, doc_id in enumerate(chunk_ids)
    ]


class TestDenseToEvidenceStep:
    def test_registered_in_builtins(self) -> None:
        step = get_step("dense_to_evidence")
        assert step.name == "dense_to_evidence"
        assert isinstance(step, DenseToEvidenceStep)

    def test_converts_dense_results_to_evidence(self) -> None:
        chunks = _make_chunks(5)
        spec = RAGAppSpec(name="test")
        ctx = build_context(
            query_text="topic 2",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=10,
        )
        ctx.dense_results = _make_dense_results(
            [c.id for c in chunks]
        )

        step = DenseToEvidenceStep()
        step.execute(ctx)

        assert len(ctx.evidence) == 5
        # Scores should match dense results
        assert ctx.evidence[0].score == 1.0
        assert ctx.evidence[1].score == 0.9
        # Retriever should be "dense_only"
        assert all(e.retriever == "dense_only" for e in ctx.evidence)
        # Ranks should be sequential
        assert [e.rank for e in ctx.evidence] == [0, 1, 2, 3, 4]
        # Strategy signal should be set
        assert all(
            e.signals.get("strategy") == "dense_dominant" for e in ctx.evidence
        )

    def test_respects_limit(self) -> None:
        chunks = _make_chunks(10)
        spec = RAGAppSpec(name="test")
        ctx = build_context(
            query_text="topic",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=3,
        )
        ctx.dense_results = _make_dense_results(
            [c.id for c in chunks]
        )

        step = DenseToEvidenceStep()
        step.execute(ctx)

        assert len(ctx.evidence) == 3

    def test_noop_with_empty_dense_results(self) -> None:
        chunks = _make_chunks(3)
        spec = RAGAppSpec(name="test")
        ctx = build_context(
            query_text="topic",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=10,
        )
        ctx.dense_results = []

        step = DenseToEvidenceStep()
        step.execute(ctx)

        assert ctx.evidence == []

    def test_skips_missing_chunks(self) -> None:
        chunks = _make_chunks(3)
        spec = RAGAppSpec(name="test")
        ctx = build_context(
            query_text="topic",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=10,
        )
        # Include a result for a non-existent chunk
        ctx.dense_results = _make_dense_results(
            ["doc_0", "nonexistent_id", "doc_2"]
        )

        step = DenseToEvidenceStep()
        step.execute(ctx)

        assert len(ctx.evidence) == 2
        assert ctx.evidence[0].chunk.id == "doc_0"
        assert ctx.evidence[1].chunk.id == "doc_2"

    def test_does_not_touch_sparse_or_fused(self) -> None:
        """DenseToEvidenceStep should not modify sparse_results or fused_results."""
        chunks = _make_chunks(3)
        spec = RAGAppSpec(name="test")
        ctx = build_context(
            query_text="topic",
            chunks=chunks,
            spec=spec,
            adaptive=AdaptiveParams(),
            limit=10,
        )
        ctx.dense_results = _make_dense_results(["doc_0", "doc_1"])

        step = DenseToEvidenceStep()
        step.execute(ctx)

        # sparse_results and fused_results should remain empty
        assert ctx.sparse_results == []
        assert ctx.fused_results == []
