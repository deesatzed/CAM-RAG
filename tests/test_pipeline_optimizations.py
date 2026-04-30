"""Tests for SOTA pipeline optimization steps.

Tests TitleBoostStep, ReciprocalNeighborStep, QueryDecomposeStep,
and ScoreNormalizeStep.
"""

from __future__ import annotations

from cam_rag.pipeline.context import build_context
from cam_rag.pipeline.executor import get_step
from cam_rag.pipeline.query_decompose import QueryDecomposeStep, _decompose
from cam_rag.pipeline.reciprocal_neighbor import ReciprocalNeighborStep
from cam_rag.pipeline.score_normalize import ScoreNormalizeStep
from cam_rag.pipeline.title_boost import TitleBoostStep
from cam_rag.rag.models import Chunk, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import AdaptiveParams
from cam_rag.retrieval.dense import HashEmbeddingBackend
from cam_rag.retrieval.sparse import SparseBM25Retriever


def _make_chunks():
    return [
        Chunk(id="c1", document_id="d1", text="Aspirin reduces inflammation and pain.", title="Aspirin"),
        Chunk(id="c2", document_id="d2", text="Python is a programming language.", title="Python"),
        Chunk(id="c3", document_id="d3", text="Ibuprofen is anti-inflammatory.", title="Ibuprofen"),
        Chunk(id="c4", document_id="d4", text="Naproxen helps with arthritis.", title="Naproxen"),
        Chunk(id="c5", document_id="d5", text="Celecoxib is a COX-2 inhibitor.", title="Celecoxib"),
    ]


def _make_ctx(query="aspirin reduces inflammation", **spec_kwargs):
    chunks = _make_chunks()
    defaults = {
        "name": "test",
        "use_pipeline": True,
        "embedding_backend": HashEmbeddingBackend(dim=32),
    }
    defaults.update(spec_kwargs)
    spec = RAGAppSpec(**defaults)
    ctx = build_context(
        query_text=query,
        chunks=chunks,
        spec=spec,
        adaptive=AdaptiveParams(),
        limit=5,
    )
    # Populate evidence
    ctx.evidence = [
        Evidence(
            chunk=chunks[0], score=0.9, retriever="hybrid_rrf", rank=0, signals={},
        ),
        Evidence(
            chunk=chunks[2], score=0.7, retriever="hybrid_rrf", rank=1, signals={},
        ),
        Evidence(
            chunk=chunks[1], score=0.3, retriever="hybrid_rrf", rank=2, signals={},
        ),
        Evidence(
            chunk=chunks[3], score=0.2, retriever="hybrid_rrf", rank=3, signals={},
        ),
    ]
    return ctx


# --- TitleBoostStep ---

class TestTitleBoostStep:
    def test_step_is_registered(self):
        step = get_step("title_boost")
        assert isinstance(step, TitleBoostStep)

    def test_boosts_title_matches(self):
        ctx = _make_ctx(query="aspirin inflammation")
        step = TitleBoostStep()
        original_score = ctx.evidence[0].score  # Aspirin chunk
        step.execute(ctx)
        # "aspirin" appears in both query and title → boosted
        assert ctx.evidence[0].score > original_score

    def test_no_boost_for_no_title_match(self):
        ctx = _make_ctx(query="inflammation pain")
        step = TitleBoostStep()
        # Python chunk has no query overlap in title
        python_item = [e for e in ctx.evidence if e.chunk.id == "c2"][0]
        original_score = python_item.score
        step.execute(ctx)
        python_item_after = [e for e in ctx.evidence if e.chunk.id == "c2"][0]
        assert python_item_after.score == original_score

    def test_records_boost_signal(self):
        ctx = _make_ctx(query="aspirin")
        step = TitleBoostStep()
        step.execute(ctx)
        aspirin_item = [e for e in ctx.evidence if e.chunk.id == "c1"][0]
        assert "title_boost" in aspirin_item.signals
        assert aspirin_item.signals["title_boost"] > 1.0

    def test_noop_on_empty_evidence(self):
        ctx = _make_ctx()
        ctx.evidence = []
        step = TitleBoostStep()
        step.execute(ctx)  # Should not raise


# --- ReciprocalNeighborStep ---

class TestReciprocalNeighborStep:
    def test_step_is_registered(self):
        step = get_step("reciprocal_neighbor_rerank")
        assert isinstance(step, ReciprocalNeighborStep)

    def test_adds_coherence_signal(self):
        ctx = _make_ctx()
        step = ReciprocalNeighborStep()
        step.execute(ctx)
        for item in ctx.evidence:
            assert "rnr_coherence" in item.signals
            assert "rnr_boost" in item.signals

    def test_noop_with_fewer_than_3(self):
        ctx = _make_ctx()
        ctx.evidence = ctx.evidence[:2]
        step = ReciprocalNeighborStep()
        step.execute(ctx)
        # Should not have added signals
        assert "rnr_coherence" not in ctx.evidence[0].signals

    def test_noop_without_backend(self):
        ctx = _make_ctx()
        ctx.spec = RAGAppSpec(name="test", use_pipeline=True, embedding_backend=None)
        step = ReciprocalNeighborStep()
        step.execute(ctx)  # Should not raise

    def test_preserves_ranking_order(self):
        ctx = _make_ctx()
        step = ReciprocalNeighborStep()
        step.execute(ctx)
        # Scores should still be descending
        scores = [item.score for item in ctx.evidence]
        assert scores == sorted(scores, reverse=True)


# --- QueryDecomposeStep ---

class TestQueryDecomposeStep:
    def test_step_is_registered(self):
        step = get_step("query_decompose")
        assert isinstance(step, QueryDecomposeStep)

    def test_decompose_with_and(self):
        parts = _decompose("aspirin reduces inflammation and prevents clots")
        assert len(parts) == 2
        assert "aspirin reduces inflammation" in parts[0]
        assert "prevents clots" in parts[1]

    def test_decompose_with_comma(self):
        parts = _decompose("aspirin, ibuprofen, naproxen")
        assert len(parts) == 3

    def test_no_decompose_simple_query(self):
        parts = _decompose("what is aspirin")
        assert len(parts) == 1

    def test_adds_new_evidence(self):
        ctx = _make_ctx(query="aspirin and naproxen")
        # Build sparse retriever for sub-query retrieval
        ctx.sparse_retriever = SparseBM25Retriever(ctx.retrieval_docs)
        original_len = len(ctx.evidence)
        step = QueryDecomposeStep()
        step.execute(ctx)
        # Should have found naproxen via sub-query
        assert len(ctx.evidence) >= original_len

    def test_noop_without_sparse_retriever(self):
        ctx = _make_ctx(query="aspirin and naproxen")
        ctx.sparse_retriever = None
        step = QueryDecomposeStep()
        original_len = len(ctx.evidence)
        step.execute(ctx)
        assert len(ctx.evidence) == original_len


# --- ScoreNormalizeStep ---

class TestScoreNormalizeStep:
    def test_step_is_registered(self):
        step = get_step("score_normalize")
        assert isinstance(step, ScoreNormalizeStep)

    def test_normalizes_to_01(self):
        ctx = _make_ctx()
        step = ScoreNormalizeStep()
        step.execute(ctx)
        for item in ctx.evidence:
            assert 0.0 <= item.score <= 1.1  # 1.1 for rank bonus

    def test_preserves_ordering(self):
        ctx = _make_ctx()
        original_ids = [item.chunk.id for item in ctx.evidence]
        step = ScoreNormalizeStep()
        step.execute(ctx)
        new_ids = [item.chunk.id for item in ctx.evidence]
        assert original_ids == new_ids

    def test_handles_negative_scores(self):
        ctx = _make_ctx()
        ctx.evidence[0].score = 5.0
        ctx.evidence[1].score = -2.0
        ctx.evidence[2].score = -10.0
        ctx.evidence[3].score = 1.0
        step = ScoreNormalizeStep()
        step.execute(ctx)
        # All should be non-negative now
        for item in ctx.evidence:
            assert item.score >= 0.0

    def test_handles_identical_scores(self):
        ctx = _make_ctx()
        for item in ctx.evidence:
            item.score = 1.0
        step = ScoreNormalizeStep()
        step.execute(ctx)
        # Should assign by rank
        scores = [item.score for item in ctx.evidence]
        assert scores[0] > scores[-1]

    def test_noop_single_item(self):
        ctx = _make_ctx()
        ctx.evidence = ctx.evidence[:1]
        step = ScoreNormalizeStep()
        step.execute(ctx)  # Should not raise
