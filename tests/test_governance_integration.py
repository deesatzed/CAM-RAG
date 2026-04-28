"""Tests for governance integration into the PipelineExecutor and query path.

Validates that:
1. Governance is disabled by default (use_governance=False)
2. Step timings are recorded in PipelineContext
3. Fitness observations are recorded when governance is enabled
4. Lifecycle transitions evaluate after enough queries
5. Latency scoring is computed from real step timings
6. Backward compatibility: executor.run() works without governance params
7. Integration with query path: _rank_chunks_pipeline with governance

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

import pytest

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import PipelineExecutor
from cam_rag.pipeline.governance import (
    EMBRYONIC_OBSERVATION_MIN,
    FitnessScore,
    FitnessTracker,
    LifecycleManager,
    TechniqueState,
)
from cam_rag.pipeline.registry import (
    ExecutionPlan,
    TechniqueDescriptor,
    TechniqueRegistry,
    compose_pipeline,
)
from cam_rag.query import (
    _get_governance,
    _rank_chunks_pipeline,
    classify_query_type,
    query,
)
from cam_rag.rag.models import Chunk, CorpusDocument
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
    defaults = {"name": "test-gov-app", "retrieval_top_k": 3}
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
    adaptive = compute_adaptive_params(
        query_type, query_text, corpus_size=len(chunks)
    )
    return build_context(query_text, chunks, spec, adaptive, limit=spec.retrieval_top_k)


def _basic_plan() -> ExecutionPlan:
    """Return a minimal 3-step plan for retrieval."""
    return ExecutionPlan(
        steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
        query_type="summary",
    )


# ---------------------------------------------------------------------------
# 1. Governance disabled by default
# ---------------------------------------------------------------------------


class TestGovernanceDisabledByDefault:
    """spec.use_governance=False means no FitnessTracker is created."""

    def test_use_governance_defaults_to_false(self):
        spec = _make_spec()
        assert spec.use_governance is False

    def test_pipeline_runs_without_governance(self):
        """The pipeline path should work normally without governance."""
        spec = _make_spec(use_pipeline=True, use_governance=False)
        docs = [
            CorpusDocument(
                id="d1",
                text="Triage scoring protocol for emergency departments.",
                source="triage.md",
            ),
            CorpusDocument(
                id="d2",
                text="Billing code reference manual.",
                source="billing.md",
            ),
        ]
        answer = query(
            question="triage scoring", documents=docs, spec=spec
        )
        assert len(answer.evidence) > 0

    def test_executor_run_without_governance_params(self):
        """executor.run(plan, ctx) with no governance params still works."""
        plan = _basic_plan()
        ctx = _make_context()
        executor = PipelineExecutor()
        result = executor.run(plan, ctx)
        assert result is ctx
        assert len(ctx.evidence) > 0
        assert ctx.steps_executed == [
            "sparse_bm25",
            "dense_vector",
            "rrf_fusion",
        ]


# ---------------------------------------------------------------------------
# 2. Step timings recorded
# ---------------------------------------------------------------------------


class TestStepTimingsRecorded:
    """After executor.run(), ctx.step_timings has entries for each step."""

    def test_timings_populated_for_each_step(self):
        plan = _basic_plan()
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)

        assert len(ctx.step_timings) == 3
        for step_name in plan.steps:
            assert step_name in ctx.step_timings

    def test_timings_are_positive_floats(self):
        plan = _basic_plan()
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)

        for step_name, timing in ctx.step_timings.items():
            assert isinstance(timing, float), (
                f"{step_name} timing is {type(timing).__name__}, expected float"
            )
            assert timing > 0.0, (
                f"{step_name} timing is {timing}, expected positive"
            )

    def test_timings_recorded_without_governance(self):
        """Timings are always recorded, even without governance enabled."""
        plan = _basic_plan()
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)

        assert "sparse_bm25" in ctx.step_timings
        assert "dense_vector" in ctx.step_timings
        assert "rrf_fusion" in ctx.step_timings

    def test_empty_plan_no_timings(self):
        plan = ExecutionPlan(steps=(), query_type="summary")
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)
        assert ctx.step_timings == {}


# ---------------------------------------------------------------------------
# 3. Fitness observations recorded
# ---------------------------------------------------------------------------


class TestFitnessObservationsRecorded:
    """With governance enabled, FitnessTracker has observations for each step."""

    def test_tracker_has_observations_for_each_step(self):
        plan = _basic_plan()
        ctx = _make_context()
        tracker = FitnessTracker()

        PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        for step_name in plan.steps:
            assert tracker.observation_count(step_name) == 1, (
                f"{step_name} should have 1 observation"
            )

    def test_tracker_fitness_is_positive(self):
        plan = _basic_plan()
        ctx = _make_context()
        tracker = FitnessTracker()

        PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        for step_name in plan.steps:
            fitness = tracker.get_fitness(step_name)
            assert fitness > 0.0, (
                f"{step_name} fitness is {fitness}, expected positive"
            )

    def test_tracker_accumulates_across_runs(self):
        """Running the pipeline twice doubles the observation count."""
        tracker = FitnessTracker()

        for _ in range(2):
            plan = _basic_plan()
            ctx = _make_context()
            PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        for step_name in plan.steps:
            assert tracker.observation_count(step_name) == 2

    def test_no_observations_when_tracker_is_none(self):
        """Without a tracker, no fitness data is recorded."""
        plan = _basic_plan()
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx, fitness_tracker=None)

        # No crash, and context still has evidence
        assert len(ctx.evidence) > 0


# ---------------------------------------------------------------------------
# 4. Lifecycle transitions evaluate
# ---------------------------------------------------------------------------


class TestLifecycleTransitions:
    """With enough queries, techniques transition states."""

    def test_techniques_start_embryonic(self):
        mgr = LifecycleManager()
        for step_name in ("sparse_bm25", "dense_vector", "rrf_fusion"):
            assert mgr.get_state(step_name) == TechniqueState.EMBRYONIC

    def test_transitions_to_active_after_threshold_queries(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            plan = _basic_plan()
            ctx = _make_context()
            PipelineExecutor().run(
                plan, ctx,
                fitness_tracker=tracker,
                lifecycle_manager=mgr,
            )

        # After enough observations, each step should be ACTIVE
        for step_name in ("sparse_bm25", "dense_vector", "rrf_fusion"):
            assert tracker.observation_count(step_name) == EMBRYONIC_OBSERVATION_MIN
            assert mgr.get_state(step_name) == TechniqueState.ACTIVE

    def test_stays_embryonic_before_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        for _ in range(EMBRYONIC_OBSERVATION_MIN - 1):
            plan = _basic_plan()
            ctx = _make_context()
            PipelineExecutor().run(
                plan, ctx,
                fitness_tracker=tracker,
                lifecycle_manager=mgr,
            )

        for step_name in ("sparse_bm25", "dense_vector", "rrf_fusion"):
            assert mgr.get_state(step_name) == TechniqueState.EMBRYONIC

    def test_lifecycle_not_evaluated_when_manager_is_none(self):
        """Without a lifecycle_manager, no transitions occur."""
        tracker = FitnessTracker()

        for _ in range(EMBRYONIC_OBSERVATION_MIN + 5):
            plan = _basic_plan()
            ctx = _make_context()
            PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        # Tracker has observations, but no lifecycle manager was applied
        assert tracker.observation_count("sparse_bm25") > 0
        # No crash -- just no state machine available to check


# ---------------------------------------------------------------------------
# 5. Latency scoring
# ---------------------------------------------------------------------------


class TestLatencyScoring:
    """Verify the latency_efficiency dimension is computed from real step timings."""

    def test_fast_steps_get_high_latency_score(self):
        """Steps that execute quickly should get latency_efficiency near 1.0."""
        plan = _basic_plan()
        ctx = _make_context()
        tracker = FitnessTracker()

        PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        # Our test steps should all complete well under 5 seconds
        for step_name in plan.steps:
            timing = ctx.step_timings[step_name]
            expected_latency_score = max(0.0, 1.0 - timing / 5.0)
            # The fitness is a weighted sum, but latency contributes positively
            # For fast steps, latency_efficiency should be very close to 1.0
            assert expected_latency_score > 0.9, (
                f"{step_name} took {timing:.4f}s, "
                f"latency_score={expected_latency_score:.4f}"
            )

    def test_latency_score_formula_consistency(self):
        """Verify that the recorded fitness is consistent with the timing data."""
        plan = _basic_plan()
        ctx = _make_context()
        tracker = FitnessTracker()

        PipelineExecutor().run(plan, ctx, fitness_tracker=tracker)

        # Compute expected overall for any step and compare with tracker
        evidence_ratio = (
            min(1.0, len(ctx.evidence) / ctx.limit)
            if ctx.limit > 0
            else 0.0
        )
        for step_name in plan.steps:
            timing = ctx.step_timings[step_name]
            latency_score = max(0.0, 1.0 - timing / 5.0)
            expected_score = FitnessScore(
                retrieval_quality=evidence_ratio,
                query_type_affinity=0.5,
                confidence_lift=0.5,
                diversity_contribution=0.5,
                latency_efficiency=latency_score,
                recency=1.0,
            )
            actual_fitness = tracker.get_fitness(step_name)
            # First observation seeds the EMA directly
            assert actual_fitness == pytest.approx(
                expected_score.overall, abs=1e-9
            ), (
                f"{step_name}: expected {expected_score.overall}, "
                f"got {actual_fitness}"
            )


# ---------------------------------------------------------------------------
# 6. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing usage patterns must still work unchanged."""

    def test_executor_run_positional_only(self):
        """executor.run(plan, ctx) with just positional args works."""
        plan = _basic_plan()
        ctx = _make_context()
        executor = PipelineExecutor()
        result = executor.run(plan, ctx)
        assert result is ctx
        assert ctx.steps_executed == [
            "sparse_bm25",
            "dense_vector",
            "rrf_fusion",
        ]
        assert len(ctx.evidence) > 0

    def test_step_timings_field_exists_on_new_context(self):
        """PipelineContext always has step_timings, defaulting to empty dict."""
        ctx = _make_context()
        assert ctx.step_timings == {}

    def test_executor_returns_context(self):
        """The executor returns the same context it was given."""
        plan = _basic_plan()
        ctx = _make_context()
        result = PipelineExecutor().run(plan, ctx)
        assert result is ctx

    def test_empty_plan_backward_compatible(self):
        plan = ExecutionPlan(steps=(), query_type="summary")
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)
        assert ctx.evidence == []
        assert ctx.steps_executed == []
        assert ctx.step_timings == {}

    def test_unknown_step_still_raises(self):
        plan = ExecutionPlan(
            steps=("nonexistent_step",), query_type="summary"
        )
        ctx = _make_context()
        with pytest.raises(KeyError, match="nonexistent_step"):
            PipelineExecutor().run(plan, ctx)


# ---------------------------------------------------------------------------
# 7. Integration with query path
# ---------------------------------------------------------------------------


class TestQueryPathIntegration:
    """_rank_chunks_pipeline with use_governance=True records governance data."""

    def test_governance_enabled_records_observations(self):
        """When use_governance=True, the governance caches are populated."""
        # Use a unique spec name to avoid cross-test pollution
        spec = _make_spec(
            name="gov-integration-test-1",
            use_pipeline=True,
            use_governance=True,
        )
        chunks = _make_chunks()
        query_type = classify_query_type("emergency triage")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage", corpus_size=len(chunks)
        )

        evidence, _expanded, _gen = _rank_chunks_pipeline(
            "emergency triage",
            chunks,
            spec,
            adaptive=adaptive,
            limit=3,
        )

        assert len(evidence) > 0

        # Verify governance caches were populated
        tracker, _mgr = _get_governance(spec)
        assert len(tracker.technique_names) > 0
        for name in tracker.technique_names:
            assert tracker.observation_count(name) >= 1

    def test_governance_disabled_no_observations(self):
        """When use_governance=False, no governance data is recorded."""
        spec = _make_spec(
            name="gov-integration-test-2",
            use_pipeline=True,
            use_governance=False,
        )
        chunks = _make_chunks()
        query_type = classify_query_type("emergency triage")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage", corpus_size=len(chunks)
        )

        evidence, _expanded, _gen = _rank_chunks_pipeline(
            "emergency triage",
            chunks,
            spec,
            adaptive=adaptive,
            limit=3,
        )

        assert len(evidence) > 0
        # No governance caches should exist for this spec
        from cam_rag.query import _fitness_trackers

        assert spec.name not in _fitness_trackers

    def test_governance_caches_reused_across_calls(self):
        """Multiple calls with the same spec reuse governance objects."""
        spec = _make_spec(
            name="gov-integration-test-3",
            use_pipeline=True,
            use_governance=True,
        )
        chunks = _make_chunks()
        query_type = classify_query_type("emergency triage")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage", corpus_size=len(chunks)
        )

        _rank_chunks_pipeline(
            "emergency triage",
            chunks,
            spec,
            adaptive=adaptive,
            limit=3,
        )
        _rank_chunks_pipeline(
            "sepsis screening criteria",
            chunks,
            spec,
            adaptive=adaptive,
            limit=3,
        )

        tracker, _mgr = _get_governance(spec)
        # Each step should have 2 observations (one per call)
        for name in tracker.technique_names:
            assert tracker.observation_count(name) >= 2

    def test_query_api_with_governance(self):
        """The top-level query() API works with governance enabled."""
        docs = [
            CorpusDocument(
                id="d1",
                text="Emergency triage scoring uses vital signs.",
                source="triage.md",
            ),
            CorpusDocument(
                id="d2",
                text="Billing code reference manual for hospitals.",
                source="billing.md",
            ),
        ]
        spec = _make_spec(
            name="gov-integration-test-4",
            use_pipeline=True,
            use_governance=True,
        )
        answer = query(
            question="triage scoring", documents=docs, spec=spec
        )
        assert len(answer.evidence) > 0

        # Verify governance data was recorded
        tracker, _mgr = _get_governance(spec)
        assert len(tracker.technique_names) > 0

    def test_full_plan_with_verification_steps(self):
        """Full plan including verification steps records all timings."""
        registry = TechniqueRegistry()
        for t in [
            TechniqueDescriptor(
                name="sparse_bm25", category="retrieval"
            ),
            TechniqueDescriptor(
                name="dense_vector", category="retrieval"
            ),
            TechniqueDescriptor(
                name="rrf_fusion",
                category="retrieval",
                requires=("sparse_bm25", "dense_vector"),
            ),
            TechniqueDescriptor(
                name="confidence_scoring", category="verification"
            ),
            TechniqueDescriptor(
                name="citation_grounding",
                category="verification",
                requires=("confidence_scoring",),
            ),
        ]:
            registry.register(t)

        plan = compose_pipeline(registry, "specification")
        ctx = _make_context()
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        PipelineExecutor().run(
            plan, ctx,
            fitness_tracker=tracker,
            lifecycle_manager=mgr,
        )

        # All steps should have timings and observations
        for step_name in plan.steps:
            assert step_name in ctx.step_timings
            assert ctx.step_timings[step_name] > 0.0
            assert tracker.observation_count(step_name) == 1
