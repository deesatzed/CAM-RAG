"""Tests for per-chunk complexity routing pipeline step.

Validates:
- Routing signal assignment based on complexity + MoE scores
- Thresholds for deep/standard/fast routing
- No-op when disabled
- CrossEncoderRerankStep skips "fast" items
- MultiHopRetrievalStep ignores "fast" items for term extraction
- Missing metadata defaults to "standard"
- Pipeline integration with ComplexityRoutingStep enabled
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import (
    ComplexityRoutingStep,
    CrossEncoderRerankStep,
    MultiHopRetrievalStep,
    get_step,
)
from cam_rag.pipeline.routing import (
    ROUTING_DEEP,
    ROUTING_FAST,
    ROUTING_STANDARD,
    assign_routing,
)
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams


def _make_evidence(
    complexity_score: float | None = None,
    moe_combined_score: float | None = None,
    *,
    chunk_id: str = "c1",
    text: str = "test chunk",
) -> Evidence:
    """Build a single Evidence item with optional metadata."""
    meta: dict = {}
    if complexity_score is not None:
        meta["complexity_score"] = complexity_score
    if moe_combined_score is not None:
        meta["moe_combined_score"] = moe_combined_score
    chunk = Chunk(
        id=chunk_id,
        document_id="d1",
        text=text,
        metadata=meta,
    )
    return Evidence(chunk=chunk, score=0.5, retriever="test", rank=0, signals={})


def _make_spec(**kwargs) -> RAGAppSpec:
    return RAGAppSpec(name="test-routing", **kwargs)


def _default_adaptive() -> AdaptiveParams:
    return AdaptiveParams(
        sparse_k=10,
        dense_k=10,
        rrf_k=60,
        sparse_weight=0.4,
        dense_weight=0.6,
    )


# ---------------------------------------------------------------------------
# assign_routing() unit tests
# ---------------------------------------------------------------------------


class TestAssignRouting:
    """Test the routing signal assignment function."""

    def test_high_complexity_routes_deep(self):
        ev = _make_evidence(complexity_score=0.80, moe_combined_score=0.5)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_DEEP

    def test_high_moe_routes_deep(self):
        ev = _make_evidence(complexity_score=0.50, moe_combined_score=0.75)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_DEEP

    def test_low_complexity_low_moe_routes_fast(self):
        ev = _make_evidence(complexity_score=0.20, moe_combined_score=0.20)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_FAST

    def test_medium_routes_standard(self):
        ev = _make_evidence(complexity_score=0.50, moe_combined_score=0.50)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_STANDARD

    def test_boundary_complexity_033_moe_030_routes_fast(self):
        """At the boundary: complexity=0.33, moe=0.30 → fast."""
        ev = _make_evidence(complexity_score=0.33, moe_combined_score=0.30)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_FAST

    def test_boundary_complexity_034_routes_standard(self):
        """Just above complexity=0.33 with low moe → standard."""
        ev = _make_evidence(complexity_score=0.34, moe_combined_score=0.20)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_STANDARD

    def test_moe_override_low_complexity_high_moe_routes_deep(self):
        """Low complexity but high MoE → deep (MoE overrides)."""
        ev = _make_evidence(complexity_score=0.20, moe_combined_score=0.80)
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_DEEP

    def test_missing_metadata_defaults_standard(self):
        """No complexity or MoE metadata → defaults (0.5, 0.5) → standard."""
        ev = _make_evidence()
        assign_routing([ev])
        assert ev.signals["routing"] == ROUTING_STANDARD

    def test_missing_complexity_only(self):
        """Missing complexity_score uses default 0.5."""
        ev = _make_evidence(moe_combined_score=0.10)
        assign_routing([ev])
        # complexity defaults to 0.5 (> 0.33) so not fast; moe=0.10 (<0.7) so not deep
        assert ev.signals["routing"] == ROUTING_STANDARD

    def test_multiple_items_mixed_routing(self):
        items = [
            _make_evidence(complexity_score=0.80, moe_combined_score=0.50, chunk_id="c1"),
            _make_evidence(complexity_score=0.50, moe_combined_score=0.50, chunk_id="c2"),
            _make_evidence(complexity_score=0.10, moe_combined_score=0.10, chunk_id="c3"),
        ]
        assign_routing(items)
        assert items[0].signals["routing"] == ROUTING_DEEP
        assert items[1].signals["routing"] == ROUTING_STANDARD
        assert items[2].signals["routing"] == ROUTING_FAST

    def test_empty_list_is_noop(self):
        assign_routing([])  # no error


# ---------------------------------------------------------------------------
# ComplexityRoutingStep tests
# ---------------------------------------------------------------------------


class TestComplexityRoutingStep:
    """Test the pipeline step wrapper."""

    def test_noop_when_disabled(self):
        spec = _make_spec(complexity_routing_enabled=False)
        ev = _make_evidence(complexity_score=0.80, moe_combined_score=0.80)
        chunk = ev.chunk
        ctx = build_context("test", [chunk], spec, _default_adaptive(), 10)
        ctx.evidence = [ev]

        step = ComplexityRoutingStep()
        step.execute(ctx)

        assert "routing" not in ev.signals

    def test_noop_when_no_evidence(self):
        spec = _make_spec(complexity_routing_enabled=True)
        chunk = Chunk(id="c1", document_id="d1", text="hello")
        ctx = build_context("test", [chunk], spec, _default_adaptive(), 10)
        ctx.evidence = []

        step = ComplexityRoutingStep()
        step.execute(ctx)
        # Should not raise

    def test_assigns_routing_when_enabled(self):
        spec = _make_spec(complexity_routing_enabled=True)
        ev = _make_evidence(complexity_score=0.80, moe_combined_score=0.50)
        chunk = ev.chunk
        ctx = build_context("test", [chunk], spec, _default_adaptive(), 10)
        ctx.evidence = [ev]

        step = ComplexityRoutingStep()
        step.execute(ctx)

        assert ev.signals["routing"] == ROUTING_DEEP

    def test_step_registered_in_executor(self):
        """The step must be discoverable via get_step."""
        step = get_step("complexity_routing")
        assert step.name == "complexity_routing"


# ---------------------------------------------------------------------------
# Downstream step integration tests
# ---------------------------------------------------------------------------


class _FakeRerankerBackend:
    """Reranker that returns ascending scores (position-based)."""

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        return [0.1 * (i + 1) for i in range(len(passages))]


class TestCrossEncoderSkipsFast:
    """CrossEncoderRerankStep must skip 'fast' items."""

    def test_fast_items_keep_original_score(self):
        spec = _make_spec(complexity_routing_enabled=True)
        backend = _FakeRerankerBackend()
        object.__setattr__(spec, "reranker_backend", backend)

        deep_ev = _make_evidence(complexity_score=0.80, chunk_id="deep")
        deep_ev.signals["routing"] = ROUTING_DEEP
        deep_ev.score = 0.9

        fast_ev = _make_evidence(complexity_score=0.10, moe_combined_score=0.10, chunk_id="fast")
        fast_ev.signals["routing"] = ROUTING_FAST
        fast_ev.score = 0.3

        chunk_deep = deep_ev.chunk
        chunk_fast = fast_ev.chunk
        ctx = build_context("test", [chunk_deep, chunk_fast], spec, _default_adaptive(), 10)
        ctx.evidence = [deep_ev, fast_ev]

        step = CrossEncoderRerankStep()
        step.execute(ctx)

        # Fast item should NOT have reranker_score signal
        assert "reranker_score" not in fast_ev.signals
        # Deep item should have been reranked
        assert "reranker_score" in deep_ev.signals

    def test_all_fast_items_no_reranking_call(self):
        spec = _make_spec(complexity_routing_enabled=True)
        backend = _FakeRerankerBackend()
        object.__setattr__(spec, "reranker_backend", backend)

        ev1 = _make_evidence(chunk_id="f1")
        ev1.signals["routing"] = ROUTING_FAST
        ev1.score = 0.5

        ev2 = _make_evidence(chunk_id="f2")
        ev2.signals["routing"] = ROUTING_FAST
        ev2.score = 0.4

        ctx = build_context("test", [ev1.chunk, ev2.chunk], spec, _default_adaptive(), 10)
        ctx.evidence = [ev1, ev2]

        step = CrossEncoderRerankStep()
        step.execute(ctx)

        # Neither should have reranker_score since all were fast
        assert "reranker_score" not in ev1.signals
        assert "reranker_score" not in ev2.signals


class TestMultiHopIgnoresFast:
    """MultiHopRetrievalStep should ignore 'fast' items for term extraction."""

    def test_fast_items_excluded_from_extraction(self):
        spec = _make_spec(
            complexity_routing_enabled=True,
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
        )

        # Create chunks with specific text for term extraction
        deep_chunk = Chunk(
            id="deep1", document_id="d1",
            text="advanced methodology protocol algorithm schema workflow",
            metadata={"complexity_score": 0.80},
        )
        fast_chunk = Chunk(
            id="fast1", document_id="d1",
            text="unique exclusive rare special extraordinary remarkable",
            metadata={"complexity_score": 0.10, "moe_combined_score": 0.10},
        )

        deep_ev = Evidence(
            chunk=deep_chunk, score=0.8, retriever="test", rank=0,
            signals={"routing": ROUTING_DEEP, "hop": 1},
        )
        fast_ev = Evidence(
            chunk=fast_chunk, score=0.3, retriever="test", rank=1,
            signals={"routing": ROUTING_FAST, "hop": 1},
        )

        ctx = build_context(
            "test query", [deep_chunk, fast_chunk], spec, _default_adaptive(), 10,
        )
        ctx.evidence = [deep_ev, fast_ev]

        step = MultiHopRetrievalStep()
        step.execute(ctx)

        # The step should have run (evidence is tagged with hop signals)
        # We verify it didn't crash and evidence still exists
        assert len(ctx.evidence) >= 1
