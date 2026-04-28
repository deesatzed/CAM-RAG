"""Tests for promoted ConfidenceScoringStep and CitationGroundingStep.

Validates that:
1. ConfidenceScoringStep produces real confidence scores from evidence
2. ConfidenceScoringStep builds citations from evidence
3. CitationGroundingStep verifies grounding against evidence
4. Both steps handle empty evidence gracefully
5. Full pipeline integration with verification steps populates all fields
6. Pipeline step results match standalone function results
7. PipelineExecutor.run() backward compatibility is preserved

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import (
    CitationGroundingStep,
    ConfidenceScoringStep,
    DenseVectorStep,
    PipelineExecutor,
    RRFFusionStep,
    SparseBM25Step,
)
from cam_rag.pipeline.registry import (
    ExecutionPlan,
    TechniqueDescriptor,
    TechniqueRegistry,
    compose_pipeline,
)
from cam_rag.query import classify_query_type
from cam_rag.rag.models import Chunk, Citation
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import compute_adaptive_params
from cam_rag.verification.confidence import score_retrieval_confidence
from cam_rag.verification.grounding import verify_citations_grounded

# ---------------------------------------------------------------------------
# Helpers
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
            section_heading=f"Heading {i}",
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
    adaptive = compute_adaptive_params(
        query_type, query_text, corpus_size=len(chunks),
    )
    return build_context(query_text, chunks, spec, adaptive, limit=spec.retrieval_top_k)


def _run_retrieval_steps(ctx: PipelineContext) -> None:
    """Run the retrieval pipeline steps to populate evidence on ctx."""
    SparseBM25Step().execute(ctx)
    DenseVectorStep().execute(ctx)
    RRFFusionStep().execute(ctx)


# ---------------------------------------------------------------------------
# 1. ConfidenceScoringStep produces real scores
# ---------------------------------------------------------------------------


class TestConfidenceScoringStepScores:
    def test_produces_float_confidence_score(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        assert len(ctx.evidence) > 0, "Precondition: evidence must exist"

        ConfidenceScoringStep().execute(ctx)

        assert isinstance(ctx.confidence_score, float)
        assert 0.0 <= ctx.confidence_score <= 1.0

    def test_confidence_details_populated(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        ConfidenceScoringStep().execute(ctx)

        assert isinstance(ctx.confidence_details, dict)
        assert "overall" in ctx.confidence_details
        assert "top_score" in ctx.confidence_details
        assert "source_agreement" in ctx.confidence_details
        assert "query_coverage" in ctx.confidence_details
        assert "evidence_count" in ctx.confidence_details

    def test_confidence_score_is_nonzero_with_evidence(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        ConfidenceScoringStep().execute(ctx)

        assert ctx.confidence_score > 0.0, (
            "Confidence should be positive when evidence exists"
        )


# ---------------------------------------------------------------------------
# 2. ConfidenceScoringStep builds citations
# ---------------------------------------------------------------------------


class TestConfidenceScoringStepCitations:
    def test_citations_populated_from_evidence(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        ConfidenceScoringStep().execute(ctx)

        assert len(ctx.citations) == len(ctx.evidence)
        assert all(isinstance(c, Citation) for c in ctx.citations)

    def test_citations_have_correct_fields(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        ConfidenceScoringStep().execute(ctx)

        for citation, evidence_item in zip(ctx.citations, ctx.evidence, strict=True):
            assert citation.source == evidence_item.chunk.source
            assert citation.document_id == evidence_item.chunk.document_id
            assert citation.title == evidence_item.chunk.title
            assert citation.section_heading == evidence_item.chunk.section_heading
            assert citation.score == evidence_item.score
            assert len(citation.excerpt) <= 240

    def test_citations_empty_before_step(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        # Before running confidence scoring, citations should be empty
        assert ctx.citations == []


# ---------------------------------------------------------------------------
# 3. CitationGroundingStep verifies grounding
# ---------------------------------------------------------------------------


class TestCitationGroundingStep:
    def test_grounded_is_bool(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        ConfidenceScoringStep().execute(ctx)

        CitationGroundingStep().execute(ctx)

        assert isinstance(ctx.grounded, bool)

    def test_grounded_true_when_citations_match_evidence(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        ConfidenceScoringStep().execute(ctx)

        CitationGroundingStep().execute(ctx)

        # Citations are built from the same evidence, so they must be grounded
        assert ctx.grounded is True

    def test_grounding_details_in_confidence_details(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        ConfidenceScoringStep().execute(ctx)

        CitationGroundingStep().execute(ctx)

        assert "grounding" in ctx.confidence_details
        grounding = ctx.confidence_details["grounding"]
        assert "grounded" in grounding
        assert "checked_citations" in grounding
        assert "matched_citations" in grounding
        assert grounding["grounded"] is True

    def test_grounding_counts_match(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        ConfidenceScoringStep().execute(ctx)

        CitationGroundingStep().execute(ctx)

        grounding = ctx.confidence_details["grounding"]
        assert grounding["checked_citations"] == len(ctx.citations)
        assert grounding["matched_citations"] == len(ctx.citations)
        assert grounding["missing_sources"] == []


# ---------------------------------------------------------------------------
# 4. Empty evidence handling
# ---------------------------------------------------------------------------


class TestEmptyEvidenceHandling:
    def test_confidence_scoring_with_no_evidence(self):
        ctx = _make_context()
        # Do not run retrieval steps -- evidence stays empty
        assert ctx.evidence == []

        ConfidenceScoringStep().execute(ctx)

        assert ctx.confidence_score == 0.0
        assert ctx.citations == []

    def test_citation_grounding_with_no_citations(self):
        ctx = _make_context()
        # No retrieval, no confidence step -- no citations
        assert ctx.citations == []

        CitationGroundingStep().execute(ctx)

        assert ctx.grounded is True  # vacuously true

    def test_citation_grounding_with_no_evidence_but_citations(self):
        ctx = _make_context()
        # Manually set citations but no evidence
        ctx.citations = [
            Citation(source="missing.md", document_id="d_missing"),
        ]
        assert ctx.evidence == []

        CitationGroundingStep().execute(ctx)

        # No evidence means vacuously true (guard condition)
        assert ctx.grounded is True


# ---------------------------------------------------------------------------
# 5. Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def _build_full_registry(self) -> TechniqueRegistry:
        registry = TechniqueRegistry()
        for t in [
            TechniqueDescriptor(name="sparse_bm25", category="retrieval"),
            TechniqueDescriptor(name="dense_vector", category="retrieval"),
            TechniqueDescriptor(
                name="rrf_fusion", category="retrieval",
                requires=("sparse_bm25", "dense_vector"),
            ),
            TechniqueDescriptor(
                name="confidence_scoring", category="verification",
            ),
            TechniqueDescriptor(
                name="citation_grounding", category="verification",
                requires=("confidence_scoring",),
            ),
        ]:
            registry.register(t)
        return registry

    def test_full_pipeline_populates_verification_fields(self):
        registry = self._build_full_registry()
        plan = compose_pipeline(registry, "specification")
        ctx = _make_context()

        executor = PipelineExecutor()
        executor.run(plan, ctx)

        # Evidence populated by retrieval steps
        assert len(ctx.evidence) > 0

        # Verification fields populated by verification steps
        assert isinstance(ctx.confidence_score, float)
        assert 0.0 <= ctx.confidence_score <= 1.0
        assert len(ctx.citations) > 0
        assert isinstance(ctx.grounded, bool)
        assert "grounding" in ctx.confidence_details

    def test_verification_steps_appear_in_execution_trace(self):
        registry = self._build_full_registry()
        plan = compose_pipeline(registry, "specification")
        ctx = _make_context()

        PipelineExecutor().run(plan, ctx)

        assert "confidence_scoring" in ctx.steps_executed
        assert "citation_grounding" in ctx.steps_executed

    def test_verification_steps_have_timings(self):
        registry = self._build_full_registry()
        plan = compose_pipeline(registry, "specification")
        ctx = _make_context()

        PipelineExecutor().run(plan, ctx)

        assert "confidence_scoring" in ctx.step_timings
        assert ctx.step_timings["confidence_scoring"] >= 0.0
        assert "citation_grounding" in ctx.step_timings
        assert ctx.step_timings["citation_grounding"] >= 0.0

    def test_full_pipeline_with_explicit_plan(self):
        """Run a plan with explicit step list including verification."""
        plan = ExecutionPlan(
            steps=(
                "sparse_bm25",
                "dense_vector",
                "rrf_fusion",
                "confidence_scoring",
                "citation_grounding",
            ),
            query_type="summary",
        )
        ctx = _make_context()

        PipelineExecutor().run(plan, ctx)

        assert ctx.confidence_score is not None
        assert ctx.confidence_score > 0.0
        assert ctx.grounded is True
        assert len(ctx.citations) == len(ctx.evidence)


# ---------------------------------------------------------------------------
# 6. Results match standalone functions
# ---------------------------------------------------------------------------


class TestResultsMatchStandalone:
    def test_confidence_score_matches_standalone(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        # Run standalone function on same evidence
        standalone_report = score_retrieval_confidence(ctx.evidence)

        # Run pipeline step
        ConfidenceScoringStep().execute(ctx)

        assert ctx.confidence_score == standalone_report.overall
        assert ctx.confidence_details["overall"] == standalone_report.overall
        assert ctx.confidence_details["top_score"] == standalone_report.top_score
        assert ctx.confidence_details["source_agreement"] == standalone_report.source_agreement
        assert ctx.confidence_details["query_coverage"] == standalone_report.query_coverage

    def test_grounding_result_matches_standalone(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)
        ConfidenceScoringStep().execute(ctx)

        # Run standalone function on same citations and evidence
        standalone_report = verify_citations_grounded(ctx.citations, ctx.evidence)

        # Run pipeline step
        CitationGroundingStep().execute(ctx)

        assert ctx.grounded == standalone_report.grounded
        grounding = ctx.confidence_details["grounding"]
        assert grounding["checked_citations"] == standalone_report.checked_citations
        assert grounding["matched_citations"] == standalone_report.matched_citations
        assert grounding["missing_sources"] == standalone_report.missing_sources

    def test_citations_match_manual_construction(self):
        ctx = _make_context()
        _run_retrieval_steps(ctx)

        # Build citations manually the same way _answer_from_evidence does
        manual_citations = [
            Citation(
                source=item.chunk.source,
                document_id=item.chunk.document_id,
                title=item.chunk.title,
                section_heading=item.chunk.section_heading,
                excerpt=item.chunk.text[:240],
                score=item.score,
            )
            for item in ctx.evidence
        ]

        # Run pipeline step
        ConfidenceScoringStep().execute(ctx)

        assert len(ctx.citations) == len(manual_citations)
        for step_cit, manual_cit in zip(ctx.citations, manual_citations, strict=True):
            assert step_cit.source == manual_cit.source
            assert step_cit.document_id == manual_cit.document_id
            assert step_cit.title == manual_cit.title
            assert step_cit.section_heading == manual_cit.section_heading
            assert step_cit.excerpt == manual_cit.excerpt
            assert step_cit.score == manual_cit.score


# ---------------------------------------------------------------------------
# 7. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_pipeline_executor_run_still_works(self):
        """PipelineExecutor.run() produces evidence with the original retrieval steps."""
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context()
        executor = PipelineExecutor()
        result_ctx = executor.run(plan, ctx)

        assert result_ctx is ctx
        assert len(ctx.evidence) > 0
        assert ctx.steps_executed == ["sparse_bm25", "dense_vector", "rrf_fusion"]

    def test_new_context_fields_default_correctly(self):
        """New verification fields start with correct defaults."""
        ctx = _make_context()

        assert ctx.citations == []
        assert ctx.confidence_score is None
        assert ctx.grounded is None
        assert ctx.confidence_details == {}

    def test_retrieval_only_plan_leaves_verification_fields_as_defaults(self):
        """When verification steps are not in the plan, their fields stay as defaults."""
        plan = ExecutionPlan(
            steps=("sparse_bm25", "dense_vector", "rrf_fusion"),
            query_type="summary",
        )
        ctx = _make_context()
        PipelineExecutor().run(plan, ctx)

        assert ctx.citations == []
        assert ctx.confidence_score is None
        assert ctx.grounded is None
        assert ctx.confidence_details == {}
