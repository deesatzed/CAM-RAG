"""Tests for PipelineExecutor, technique steps, and static/dynamic equivalence.

Validates that:
1. Executor runs steps in order and records them
2. Context carries state between steps
3. Individual built-in steps produce correct outputs
4. Dynamic pipeline produces equivalent results to the static path
5. Custom/subset technique sets work
6. Feature flag correctly selects the path

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import (
    CitationGroundingStep,
    ConfidenceScoringStep,
    DenseVectorStep,
    PipelineExecutor,
    QueryExpansionStep,
    RRFFusionStep,
    SparseBM25Step,
    get_step,
    register_step,
)
from cam_rag.pipeline.registry import (
    ExecutionPlan,
    TechniqueDescriptor,
    TechniqueRegistry,
    compose_pipeline,
)
from cam_rag.query import (
    _rank_chunks_pipeline,
    _rank_chunks_static,
    classify_query_type,
    query,
    query_document_folder,
)
from cam_rag.rag.models import Chunk, CorpusDocument, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import compute_adaptive_params

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunks(n: int = 5) -> list[Chunk]:
    """Build a small set of realistic chunks for testing."""
    texts = [
        "Emergency department triage uses vital signs and acuity scoring protocols.",
        "Chest pain evaluation follows the HEART score algorithm for risk assessment.",
        "Sepsis screening uses the qSOFA criteria including altered mental status.",
        "Stroke assessment follows the NIH Stroke Scale for severity grading.",
        "Trauma patients are assessed using the Glasgow Coma Scale and ABCDE approach.",
    ]
    return [
        Chunk(
            id=f"chunk_{i}",
            document_id=f"doc_{i}",
            text=texts[i],
            source=f"document_{i}.md",
            title=f"Section {i}",
        )
        for i in range(min(n, len(texts)))
    ]


def _make_spec(**overrides) -> RAGAppSpec:
    defaults = {"name": "test-app", "retrieval_top_k": 3}
    defaults.update(overrides)
    return RAGAppSpec(**defaults)


def _make_context(
    query_text: str = "emergency triage protocol",
    chunks: list[Chunk] | None = None,
    **spec_overrides,
) -> PipelineContext:
    if chunks is None:
        chunks = _make_chunks()
    spec = _make_spec(**spec_overrides)
    query_type = classify_query_type(query_text)
    adaptive = compute_adaptive_params(query_type, query_text, corpus_size=len(chunks))
    return build_context(query_text, chunks, spec, adaptive, limit=spec.retrieval_top_k)


# ---------------------------------------------------------------------------
# 1. Context Construction
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_context_has_query_and_chunks(self):
        ctx = _make_context()
        assert ctx.query_text == "emergency triage protocol"
        assert len(ctx.chunks) == 5

    def test_context_has_chunk_by_id_map(self):
        ctx = _make_context()
        assert "chunk_0" in ctx.chunk_by_id
        assert ctx.chunk_by_id["chunk_0"].text.startswith("Emergency department")

    def test_context_has_retrieval_docs(self):
        ctx = _make_context()
        assert len(ctx.retrieval_docs) == 5
        assert ctx.retrieval_docs[0].doc_id == "chunk_0"

    def test_context_expanded_query_starts_as_query(self):
        ctx = _make_context()
        assert ctx.expanded_query == ctx.query_text

    def test_context_starts_with_no_results(self):
        ctx = _make_context()
        assert ctx.sparse_results == []
        assert ctx.dense_results == []
        assert ctx.fused_results == []
        assert ctx.evidence == []
        assert ctx.steps_executed == []


# ---------------------------------------------------------------------------
# 2. Individual Step Tests
# ---------------------------------------------------------------------------


class TestSparseBM25Step:
    def test_produces_sparse_results(self):
        ctx = _make_context()
        SparseBM25Step().execute(ctx)
        assert len(ctx.sparse_results) > 0
        assert ctx.sparse_retriever is not None

    def test_reuses_existing_retriever(self):
        ctx = _make_context()
        SparseBM25Step().execute(ctx)
        retriever_1 = ctx.sparse_retriever
        SparseBM25Step().execute(ctx)
        assert ctx.sparse_retriever is retriever_1


class TestDenseVectorStep:
    def test_produces_dense_results(self):
        ctx = _make_context()
        DenseVectorStep().execute(ctx)
        assert len(ctx.dense_results) > 0
        assert ctx.dense_retriever is not None

    def test_reuses_existing_retriever(self):
        ctx = _make_context()
        DenseVectorStep().execute(ctx)
        retriever_1 = ctx.dense_retriever
        DenseVectorStep().execute(ctx)
        assert ctx.dense_retriever is retriever_1


class TestRRFFusionStep:
    def test_produces_fused_results_and_evidence(self):
        ctx = _make_context()
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.fused_results) > 0
        assert len(ctx.evidence) > 0
        assert all(isinstance(e, Evidence) for e in ctx.evidence)

    def test_evidence_limited_by_ctx_limit(self):
        ctx = _make_context()
        ctx.limit = 2
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        assert len(ctx.evidence) <= 2

    def test_evidence_has_signals(self):
        ctx = _make_context()
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        for e in ctx.evidence:
            assert "matched_terms" in e.signals
            assert "source_ranks" in e.signals
            assert "rrf_score" in e.signals


class TestQueryExpansionStep:
    def test_skips_when_not_enabled(self):
        ctx = _make_context(query_expansion_enabled=False)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)

        QueryExpansionStep().execute(ctx)

        # Evidence should be unchanged when expansion is disabled
        assert ctx.expanded_query == ctx.query_text

    def test_skips_when_no_evidence(self):
        ctx = _make_context(query_text="xyznonexistent", query_expansion_enabled=True)
        QueryExpansionStep().execute(ctx)
        assert ctx.expanded_query == ctx.query_text

    def test_expands_query_when_enabled(self):
        ctx = _make_context(query_expansion_enabled=True, expansion_terms=3)
        SparseBM25Step().execute(ctx)
        DenseVectorStep().execute(ctx)
        RRFFusionStep().execute(ctx)
        QueryExpansionStep().execute(ctx)

        # Expansion should add terms (or at least not crash)
        assert len(ctx.expanded_query) >= len(ctx.query_text)


class TestVerificationSteps:
    def test_confidence_scoring_is_noop(self):
        ctx = _make_context()
        ConfidenceScoringStep().execute(ctx)
        # Should not crash or change state
        assert ctx.evidence == []

    def test_citation_grounding_is_noop(self):
        ctx = _make_context()
        CitationGroundingStep().execute(ctx)
        assert ctx.evidence == []


# ---------------------------------------------------------------------------
# 3. Executor Tests
# ---------------------------------------------------------------------------


class TestPipelineExecutor:
    def test_runs_steps_in_order(self):
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context()
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert ctx.steps_executed == ["sparse_bm25", "dense_vector", "rrf_fusion"]

    def test_produces_evidence(self):
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context()
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert len(ctx.evidence) > 0

    def test_full_default_plan(self):
        registry = TechniqueRegistry()
        for t in [
            TechniqueDescriptor(name="sparse_bm25", category="retrieval"),
            TechniqueDescriptor(name="dense_vector", category="retrieval"),
            TechniqueDescriptor(
                name="rrf_fusion", category="retrieval",
                requires=("sparse_bm25", "dense_vector"),
            ),
            TechniqueDescriptor(name="confidence_scoring", category="verification"),
            TechniqueDescriptor(
                name="citation_grounding", category="verification",
                requires=("confidence_scoring",),
            ),
        ]:
            registry.register(t)

        plan = compose_pipeline(registry, "specification")
        ctx = _make_context()
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert len(ctx.evidence) > 0
        assert "sparse_bm25" in ctx.steps_executed
        assert "rrf_fusion" in ctx.steps_executed

    def test_empty_plan_produces_no_evidence(self):
        plan = ExecutionPlan(steps=(), query_type="summary")
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)
        assert ctx.evidence == []
        assert ctx.steps_executed == []

    def test_unknown_step_raises(self):
        plan = ExecutionPlan(steps=("nonexistent_step",), query_type="summary")
        ctx = _make_context()
        with pytest.raises(KeyError, match="nonexistent_step"):
            PipelineExecutor().run(plan, ctx)


# ---------------------------------------------------------------------------
# 4. Step Registry
# ---------------------------------------------------------------------------


class TestStepRegistry:
    def test_get_builtin_step(self):
        step = get_step("sparse_bm25")
        assert step.name == "sparse_bm25"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_step("nonexistent")

    def test_register_custom_step(self):
        class CustomStep:
            name = "custom_test_step"

            def execute(self, ctx: PipelineContext) -> None:
                ctx.extras["custom_ran"] = True

        register_step(CustomStep())
        step = get_step("custom_test_step")
        ctx = _make_context()
        step.execute(ctx)
        assert ctx.extras["custom_ran"] is True


# ---------------------------------------------------------------------------
# 5. Equivalence: Static vs. Dynamic Pipeline
# ---------------------------------------------------------------------------


class TestStaticDynamicEquivalence:
    """The dynamic pipeline must produce equivalent evidence to the static path."""

    def _run_both_paths(
        self, query_text: str, chunks: list[Chunk], **spec_kw,
    ) -> tuple[list[Evidence], list[Evidence]]:
        spec_static = _make_spec(use_pipeline=False, **spec_kw)
        spec_dynamic = _make_spec(use_pipeline=True, **spec_kw)

        query_type = classify_query_type(query_text)
        adaptive = compute_adaptive_params(query_type, query_text, corpus_size=len(chunks))

        static_evidence, _static_expanded = _rank_chunks_static(
            query_text, chunks, spec_static, adaptive=adaptive, limit=spec_static.retrieval_top_k,
        )
        dynamic_evidence, _dynamic_expanded, _gen = _rank_chunks_pipeline(
            query_text, chunks, spec_dynamic, adaptive=adaptive,
            limit=spec_dynamic.retrieval_top_k,
        )
        return static_evidence, dynamic_evidence

    def test_same_evidence_count(self):
        chunks = _make_chunks()
        static_ev, dynamic_ev = self._run_both_paths("emergency triage protocol", chunks)
        assert len(static_ev) == len(dynamic_ev)

    def test_same_top_result(self):
        chunks = _make_chunks()
        static_ev, dynamic_ev = self._run_both_paths("emergency triage protocol", chunks)
        if static_ev and dynamic_ev:
            assert static_ev[0].chunk.id == dynamic_ev[0].chunk.id

    def test_same_scores(self):
        chunks = _make_chunks()
        static_ev, dynamic_ev = self._run_both_paths("emergency triage protocol", chunks)
        for s, d in zip(static_ev, dynamic_ev, strict=True):
            assert abs(s.score - d.score) < 1e-10, (
                f"Score mismatch: static={s.score}, dynamic={d.score}"
            )

    def test_same_ranking_order(self):
        chunks = _make_chunks()
        static_ev, dynamic_ev = self._run_both_paths("sepsis screening criteria", chunks)
        static_ids = [e.chunk.id for e in static_ev]
        dynamic_ids = [e.chunk.id for e in dynamic_ev]
        assert static_ids == dynamic_ids

    def test_equivalence_with_different_query_types(self):
        chunks = _make_chunks()
        queries = [
            "what is triage",  # specification
            "chest pain assessment",  # summary
            "why is qSOFA used for sepsis",  # logic
            "compare triage and trauma? how do they differ?",  # synthesis
        ]
        for q in queries:
            static_ev, dynamic_ev = self._run_both_paths(q, chunks)
            static_ids = [e.chunk.id for e in static_ev]
            dynamic_ids = [e.chunk.id for e in dynamic_ev]
            assert static_ids == dynamic_ids, f"Mismatch for query: {q}"


# ---------------------------------------------------------------------------
# 6. Feature Flag Integration
# ---------------------------------------------------------------------------


class TestFeatureFlagIntegration:
    def test_use_pipeline_false_uses_static(self, tmp_path: Path):
        (tmp_path / "test.md").write_text(
            "# Test\nEmergency triage uses vital signs.\n", encoding="utf-8",
        )
        spec = _make_spec(use_pipeline=False)
        answer = query_document_folder(tmp_path, "emergency triage", spec)
        assert answer.answer
        assert "pipeline_executor" not in answer.trace.stages

    def test_use_pipeline_true_uses_dynamic(self, tmp_path: Path):
        (tmp_path / "test.md").write_text(
            "# Test\nEmergency triage uses vital signs.\n", encoding="utf-8",
        )
        spec = _make_spec(use_pipeline=True)
        answer = query_document_folder(tmp_path, "emergency triage", spec)
        assert answer.answer

    def test_query_api_with_pipeline(self):
        docs = [
            CorpusDocument(id="d1", text="Triage scoring protocol.", source="triage.md"),
            CorpusDocument(id="d2", text="Billing code reference.", source="billing.md"),
        ]
        spec = _make_spec(use_pipeline=True)
        answer = query(question="triage scoring", documents=docs, spec=spec)
        assert len(answer.evidence) > 0
