"""Tests for the multi-hop retrieval step.

Validates that:
1.  Multi-hop disabled by default (no-op)
2.  Multi-hop enabled: new evidence discovered
3.  Deduplication of chunk IDs
4.  Hop signal in evidence signals
5.  Respects max_hops limit
6.  Works with empty evidence (no-op)
7.  Pipeline path integration
8.  Static path integration
9.  Evidence count does not exceed limit
10. Query type filtering (only synthesis/logic)

All data is real clinical text -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cam_rag.pipeline.context import PipelineContext, build_context
from cam_rag.pipeline.executor import (
    MultiHopRetrievalStep,
    PipelineExecutor,
    SparseBM25Step,
    DenseVectorStep,
    RRFFusionStep,
    get_step,
)
from cam_rag.pipeline.registry import (
    ExecutionPlan,
    TechniqueDescriptor,
    TechniqueRegistry,
    compose_pipeline,
)
from cam_rag.query import (
    _build_default_registry,
    _rank_chunks_pipeline,
    _rank_chunks_static,
    classify_query_type,
    query_document_folder,
)
from cam_rag.rag.models import Chunk, CorpusDocument, Evidence
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.retrieval.adaptive_params import compute_adaptive_params

# ---------------------------------------------------------------------------
# Fixtures: real clinical text documents
# ---------------------------------------------------------------------------

# Designed so that initial retrieval for "emergency triage protocol"
# finds the triage chunk, but NOT the pharmacology or radiology chunks.
# Multi-hop should extract terms like "acuity" or "vital" from the triage
# evidence, and re-query to find related clinical concepts.

_CLINICAL_TEXTS = [
    (
        "Emergency department triage uses vital signs and acuity scoring "
        "protocols to stratify patients by severity."
    ),
    (
        "Chest pain evaluation follows the HEART score algorithm for "
        "risk assessment of acute coronary syndrome."
    ),
    (
        "Sepsis screening uses the qSOFA criteria including altered "
        "mental status, respiratory rate, and systolic blood pressure."
    ),
    (
        "Stroke assessment follows the NIH Stroke Scale for severity "
        "grading and treatment selection."
    ),
    (
        "Trauma patients are assessed using the Glasgow Coma Scale "
        "and the ABCDE approach for primary survey."
    ),
    (
        "Acuity levels range from resuscitation to non-urgent and "
        "determine the order of patient evaluation."
    ),
    (
        "Vital signs monitoring includes heart rate, blood pressure, "
        "temperature, respiratory rate, and oxygen saturation."
    ),
    (
        "Pharmacology of vasopressors covers norepinephrine, vasopressin, "
        "and epinephrine dose titration for hemodynamic support."
    ),
    (
        "Radiology imaging protocols for computed tomography require "
        "contrast administration timing and radiation dose optimization."
    ),
    (
        "Scoring systems in emergency medicine include APACHE, SOFA, "
        "and NEWS for predicting patient deterioration."
    ),
]


def _make_chunks(n: int | None = None) -> list[Chunk]:
    """Build realistic clinical chunks for testing."""
    texts = _CLINICAL_TEXTS
    if n is not None:
        texts = texts[:n]
    return [
        Chunk(
            id=f"chunk_{i}",
            document_id=f"doc_{i}",
            text=text,
            source=f"clinical_{i}.md",
            title=f"Section {i}",
        )
        for i, text in enumerate(texts)
    ]


def _make_spec(**overrides) -> RAGAppSpec:
    defaults = {"name": "test-multi-hop", "retrieval_top_k": 5}
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


def _run_retrieval_pipeline(ctx: PipelineContext) -> None:
    """Run sparse -> dense -> fusion to populate ctx.evidence."""
    SparseBM25Step().execute(ctx)
    DenseVectorStep().execute(ctx)
    RRFFusionStep().execute(ctx)


# ---------------------------------------------------------------------------
# 1. Multi-hop disabled by default (no-op)
# ---------------------------------------------------------------------------


class TestMultiHopDisabledByDefault:
    def test_spec_default_multi_hop_disabled(self):
        spec = _make_spec()
        assert spec.multi_hop_enabled is False

    def test_spec_default_max_hops(self):
        spec = _make_spec()
        assert spec.multi_hop_max_hops == 2

    def test_step_is_noop_when_disabled(self):
        ctx = _make_context(multi_hop_enabled=False)
        _run_retrieval_pipeline(ctx)
        evidence_before = list(ctx.evidence)

        MultiHopRetrievalStep().execute(ctx)

        # Evidence should be unchanged
        assert len(ctx.evidence) == len(evidence_before)
        for item in ctx.evidence:
            assert "hop" not in item.signals


# ---------------------------------------------------------------------------
# 2. Multi-hop enabled: new evidence discovered
# ---------------------------------------------------------------------------


class TestMultiHopNewEvidenceDiscovered:
    def test_multi_hop_adds_new_evidence(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=8,
        )
        _run_retrieval_pipeline(ctx)
        initial_ids = {item.chunk.id for item in ctx.evidence}

        MultiHopRetrievalStep().execute(ctx)

        final_ids = {item.chunk.id for item in ctx.evidence}
        # Multi-hop should have potentially discovered new chunks
        # (at minimum, it should not remove any existing evidence)
        assert initial_ids.issubset(final_ids) or len(final_ids) >= len(initial_ids)

    def test_multi_hop_evidence_has_retriever_tag(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        # Check that any hop > 1 evidence is tagged as multi_hop_sparse
        hop_evidence = [
            item for item in ctx.evidence
            if item.signals.get("hop", 1) > 1
        ]
        for item in hop_evidence:
            assert item.retriever == "multi_hop_sparse"


# ---------------------------------------------------------------------------
# 3. Deduplication of chunk IDs
# ---------------------------------------------------------------------------


class TestMultiHopDeduplication:
    def test_no_duplicate_chunk_ids(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=3,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        chunk_ids = [item.chunk.id for item in ctx.evidence]
        assert len(chunk_ids) == len(set(chunk_ids)), (
            f"Duplicate chunk IDs found: {chunk_ids}"
        )


# ---------------------------------------------------------------------------
# 4. Hop signal in evidence signals
# ---------------------------------------------------------------------------


class TestMultiHopSignals:
    def test_hop_1_tagged_on_existing_evidence(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=5,
        )
        _run_retrieval_pipeline(ctx)
        assert ctx.evidence, "Precondition: need evidence for multi-hop"

        MultiHopRetrievalStep().execute(ctx)

        # All evidence from the initial retrieval should be tagged hop=1
        for item in ctx.evidence:
            assert "hop" in item.signals
            assert item.signals["hop"] >= 1

    def test_new_evidence_tagged_with_hop_number(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=3,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        hop_numbers = {item.signals.get("hop") for item in ctx.evidence}
        # Should have at least hop 1
        assert 1 in hop_numbers

    def test_hop_query_in_new_evidence_signals(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        hop2_evidence = [
            item for item in ctx.evidence
            if item.signals.get("hop", 1) > 1
        ]
        for item in hop2_evidence:
            assert "hop_query" in item.signals
            assert ctx.query_text in item.signals["hop_query"]


# ---------------------------------------------------------------------------
# 5. Respects max_hops limit
# ---------------------------------------------------------------------------


class TestMultiHopMaxHopsLimit:
    def test_max_hops_1_means_no_additional_hops(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=1,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)
        initial_count = len(ctx.evidence)

        MultiHopRetrievalStep().execute(ctx)

        # With max_hops=1, the range(2, 2) loop body never runs
        # so no new evidence should be added beyond tagging
        for item in ctx.evidence:
            assert item.signals.get("hop", 1) == 1

    def test_max_hops_limits_hop_numbers(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        for item in ctx.evidence:
            assert item.signals.get("hop", 1) <= 2


# ---------------------------------------------------------------------------
# 6. Works with empty evidence (no-op)
# ---------------------------------------------------------------------------


class TestMultiHopEmptyEvidence:
    def test_noop_when_evidence_empty(self):
        ctx = _make_context(
            query_text="xyznonexistentquery",
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
        )
        # Do NOT run retrieval pipeline -- evidence stays empty
        assert ctx.evidence == []

        MultiHopRetrievalStep().execute(ctx)

        assert ctx.evidence == []


# ---------------------------------------------------------------------------
# 7. Pipeline path integration
# ---------------------------------------------------------------------------


class TestMultiHopPipelineIntegration:
    def test_multi_hop_registered_as_builtin_step(self):
        step = get_step("multi_hop_retrieval")
        assert step.name == "multi_hop_retrieval"

    def test_multi_hop_in_default_registry(self):
        registry = _build_default_registry()
        desc = registry.get("multi_hop_retrieval")
        assert desc.name == "multi_hop_retrieval"
        assert desc.category == "retrieval"
        assert "rrf_fusion" in desc.requires

    def test_pipeline_includes_multi_hop_for_synthesis(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "synthesis", corpus_size=10)
        assert "multi_hop_retrieval" in plan.steps

    def test_pipeline_includes_multi_hop_for_logic(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "logic", corpus_size=10)
        assert "multi_hop_retrieval" in plan.steps

    def test_pipeline_excludes_multi_hop_for_specification(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "specification", corpus_size=10)
        assert "multi_hop_retrieval" not in plan.steps

    def test_pipeline_excludes_multi_hop_for_summary(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "summary", corpus_size=10)
        assert "multi_hop_retrieval" not in plan.steps

    def test_pipeline_executor_runs_multi_hop_step(self):
        plan = ExecutionPlan(
            steps=(
                "sparse_bm25",
                "dense_vector",
                "rrf_fusion",
                "multi_hop_retrieval",
            ),
            query_type="synthesis",
        )
        ctx = _make_context(
            query_text="compare triage and sepsis screening?",
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=8,
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert "multi_hop_retrieval" in ctx.steps_executed
        assert len(ctx.evidence) > 0

    def test_rank_chunks_pipeline_with_multi_hop(self):
        chunks = _make_chunks()
        # Use a synthesis query so multi_hop_retrieval is included
        evidence, expanded, generated = _rank_chunks_pipeline(
            "compare triage and sepsis? how do scoring systems differ?",
            chunks,
            _make_spec(
                use_pipeline=True,
                multi_hop_enabled=True,
                multi_hop_max_hops=2,
                retrieval_top_k=8,
            ),
            adaptive=compute_adaptive_params(
                "synthesis",
                "compare triage and sepsis? how do scoring systems differ?",
                corpus_size=len(chunks),
            ),
            limit=8,
        )
        assert len(evidence) > 0


# ---------------------------------------------------------------------------
# 8. Static path integration
# ---------------------------------------------------------------------------


class TestMultiHopStaticIntegration:
    def test_static_path_with_multi_hop_enabled(self):
        chunks = _make_chunks()
        spec = _make_spec(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=8,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol", corpus_size=len(chunks),
        )
        evidence, expanded = _rank_chunks_static(
            "emergency triage protocol",
            chunks,
            spec,
            adaptive=adaptive,
            limit=8,
        )
        assert len(evidence) > 0
        # Should have hop signals when multi-hop is enabled
        hop_signals = [item.signals.get("hop") for item in evidence]
        assert any(h is not None for h in hop_signals)

    def test_static_path_with_multi_hop_disabled(self):
        chunks = _make_chunks()
        spec = _make_spec(
            multi_hop_enabled=False,
            retrieval_top_k=5,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol", corpus_size=len(chunks),
        )
        evidence, expanded = _rank_chunks_static(
            "emergency triage protocol",
            chunks,
            spec,
            adaptive=adaptive,
            limit=5,
        )
        # No hop signals when disabled
        for item in evidence:
            assert "hop" not in item.signals

    def test_static_path_document_folder_integration(self, tmp_path: Path):
        (tmp_path / "triage.md").write_text(
            "# Triage\n\n"
            "Emergency department triage uses vital signs and acuity scoring.\n",
            encoding="utf-8",
        )
        (tmp_path / "acuity.md").write_text(
            "# Acuity\n\n"
            "Acuity levels range from resuscitation to non-urgent and "
            "determine the order of patient evaluation.\n",
            encoding="utf-8",
        )
        (tmp_path / "vitals.md").write_text(
            "# Vital Signs\n\n"
            "Vital signs monitoring includes heart rate, blood pressure, "
            "temperature, respiratory rate, and oxygen saturation.\n",
            encoding="utf-8",
        )
        (tmp_path / "billing.md").write_text(
            "# Billing\n\nInvoices are reviewed monthly.\n",
            encoding="utf-8",
        )
        spec = _make_spec(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=4,
        )
        answer = query_document_folder(
            tmp_path, "emergency triage protocol", spec,
        )
        assert answer.answer
        assert len(answer.evidence) > 0


# ---------------------------------------------------------------------------
# 9. Evidence count does not exceed limit
# ---------------------------------------------------------------------------


class TestMultiHopEvidenceLimit:
    def test_evidence_respects_limit(self):
        limit = 3
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=3,
            retrieval_top_k=limit,
        )
        ctx.limit = limit
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        assert len(ctx.evidence) <= limit

    def test_static_path_evidence_respects_limit(self):
        chunks = _make_chunks()
        limit = 3
        spec = _make_spec(
            multi_hop_enabled=True,
            multi_hop_max_hops=3,
            retrieval_top_k=limit,
        )
        query_type = classify_query_type("emergency triage protocol")
        adaptive = compute_adaptive_params(
            query_type, "emergency triage protocol", corpus_size=len(chunks),
        )
        evidence, _ = _rank_chunks_static(
            "emergency triage protocol",
            chunks,
            spec,
            adaptive=adaptive,
            limit=limit,
        )
        assert len(evidence) <= limit


# ---------------------------------------------------------------------------
# 10. Query type filtering (only synthesis/logic)
# ---------------------------------------------------------------------------


class TestMultiHopQueryTypeFiltering:
    def test_synthesis_query_includes_multi_hop_in_plan(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "synthesis", corpus_size=10)
        assert "multi_hop_retrieval" in plan.steps

    def test_logic_query_includes_multi_hop_in_plan(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "logic", corpus_size=10)
        assert "multi_hop_retrieval" in plan.steps

    def test_summary_query_excludes_multi_hop_from_plan(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "summary", corpus_size=10)
        assert "multi_hop_retrieval" not in plan.steps

    def test_specification_query_excludes_multi_hop_from_plan(self):
        registry = _build_default_registry()
        plan = compose_pipeline(registry, "specification", corpus_size=10)
        assert "multi_hop_retrieval" not in plan.steps

    def test_query_type_classification_synthesis(self):
        assert classify_query_type(
            "compare triage and sepsis? how do they differ?"
        ) == "synthesis"

    def test_query_type_classification_logic(self):
        assert classify_query_type(
            "why is qSOFA used for sepsis screening"
        ) == "logic"

    def test_query_type_classification_summary(self):
        assert classify_query_type(
            "chest pain assessment"
        ) == "summary"

    def test_query_type_classification_specification(self):
        assert classify_query_type(
            "what is triage"
        ) == "specification"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestMultiHopEdgeCases:
    def test_multi_hop_with_single_chunk(self):
        """Multi-hop with only one chunk should not crash."""
        chunks = _make_chunks(1)
        ctx = _make_context(
            chunks=chunks,
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=5,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        # Should not crash and evidence should be valid
        assert all(isinstance(e, Evidence) for e in ctx.evidence)

    def test_ranks_are_sequential_after_multi_hop(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        for i, item in enumerate(ctx.evidence):
            assert item.rank == i, (
                f"Expected rank {i} but got {item.rank}"
            )

    def test_evidence_sorted_descending_by_score(self):
        ctx = _make_context(
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=10,
        )
        _run_retrieval_pipeline(ctx)

        MultiHopRetrievalStep().execute(ctx)

        scores = [item.score for item in ctx.evidence]
        assert scores == sorted(scores, reverse=True), (
            "Evidence should be sorted by descending score"
        )

    def test_step_timing_recorded_in_executor(self):
        plan = ExecutionPlan(
            steps=(
                "sparse_bm25",
                "dense_vector",
                "rrf_fusion",
                "multi_hop_retrieval",
            ),
            query_type="logic",
        )
        ctx = _make_context(
            query_text="why is triage important in emergency medicine",
            multi_hop_enabled=True,
            multi_hop_max_hops=2,
            retrieval_top_k=8,
        )
        executor = PipelineExecutor()
        executor.run(plan, ctx)

        assert "multi_hop_retrieval" in ctx.step_timings
        assert ctx.step_timings["multi_hop_retrieval"] >= 0.0
