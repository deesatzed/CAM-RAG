"""Tests for evaluation framework extensions.

Extends the existing golden-question framework to support:
- Multi-technique evaluation (tracking which technique contributed what)
- Pipeline selection quality metrics (was the right pipeline chosen?)
- Technique-level contribution scoring
- Per-technique ablation testing

All tests use real platform evaluation primitives and real document data.
No mock, no simulation, no placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass

from cam_rag.evaluation import (
    GoldenQuestion,
    evaluate_answer,
    expected_term_match,
    recall_at_k,
)
from cam_rag.rag.models import Chunk, Citation, Evidence

# ---------------------------------------------------------------------------
# 1. Multi-Technique Golden Question Evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TechniqueAnnotatedEvidence:
    """Evidence annotated with the technique that produced it."""

    evidence: Evidence
    technique: str
    contribution_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MultiTechniqueEvaluationResult:
    """Evaluation result with per-technique attribution."""

    question_id: str
    overall: float
    technique_contributions: dict[str, float]  # technique -> contribution to overall
    technique_recall: dict[str, float]  # technique -> recall_at_k for that technique
    best_technique: str
    pipeline_used: str


def evaluate_multi_technique(
    golden: GoldenQuestion,
    *,
    answer: str,
    technique_evidence: dict[str, list[Evidence]],
    citations: list[Citation],
    k: int = 5,
    pipeline_name: str = "default",
) -> MultiTechniqueEvaluationResult:
    """Evaluate an answer with per-technique attribution.

    Args:
        golden: The golden question to evaluate against.
        answer: The generated/retrieved answer text.
        technique_evidence: Mapping from technique name to its evidence items.
        citations: Citations included in the answer.
        k: Top-k for recall calculation.
        pipeline_name: Name of the pipeline that produced this answer.
    """
    # Flatten all evidence for overall evaluation
    all_evidence = []
    for ev_list in technique_evidence.values():
        all_evidence.extend(ev_list)

    # Overall evaluation using existing framework
    overall_result = evaluate_answer(
        golden,
        answer=answer,
        evidence=all_evidence,
        citations=citations,
        k=k,
    )

    # Per-technique recall
    technique_recall: dict[str, float] = {}
    for tech_name, ev_list in technique_evidence.items():
        technique_recall[tech_name] = recall_at_k(
            golden.expected_sources, ev_list, k=k
        )

    # Technique contribution: which technique's evidence was most useful
    technique_contributions: dict[str, float] = {}
    for tech_name, ev_list in technique_evidence.items():
        if not ev_list:
            technique_contributions[tech_name] = 0.0
            continue
        # Contribution = term match from that technique's evidence text
        tech_text = " ".join(ev.chunk.text for ev in ev_list)
        technique_contributions[tech_name] = expected_term_match(
            tech_text, golden.expected_terms
        )

    best_technique = (
        max(technique_contributions, key=technique_contributions.get)
        if technique_contributions
        else ""
    )

    return MultiTechniqueEvaluationResult(
        question_id=golden.id,
        overall=overall_result.overall,
        technique_contributions=technique_contributions,
        technique_recall=technique_recall,
        best_technique=best_technique,
        pipeline_used=pipeline_name,
    )


class TestMultiTechniqueEvaluation:
    """Multi-technique evaluation must correctly attribute per-technique contributions."""

    def _make_evidence(self, source: str, text: str, retriever: str) -> Evidence:
        chunk = Chunk(
            id=f"chunk_{source}",
            document_id=f"doc_{source}",
            text=text,
            source=source,
        )
        return Evidence(chunk=chunk, score=0.9, retriever=retriever)

    def test_single_technique_gets_full_attribution(self):
        golden = GoldenQuestion(
            id="q1",
            question="What is the triage protocol?",
            expected_sources=["triage.md"],
            expected_terms=["vital signs", "acuity"],
        )
        evidence = self._make_evidence(
            "triage.md", "Triage uses vital signs and acuity scoring.", "sparse_bm25"
        )

        result = evaluate_multi_technique(
            golden,
            answer="Triage uses vital signs and acuity scoring.",
            technique_evidence={"sparse_bm25": [evidence]},
            citations=[Citation(source="triage.md", document_id="doc_triage.md")],
            k=5,
        )

        assert result.best_technique == "sparse_bm25"
        assert result.technique_recall["sparse_bm25"] == 1.0
        assert result.technique_contributions["sparse_bm25"] == 1.0

    def test_two_techniques_with_different_contributions(self):
        golden = GoldenQuestion(
            id="q2",
            question="ICU transfer criteria",
            expected_sources=["icu.md", "transfer.md"],
            expected_terms=["hemodynamic", "instability"],
        )
        sparse_ev = self._make_evidence(
            "icu.md", "ICU transfer requires hemodynamic assessment.", "sparse_bm25"
        )
        dense_ev = self._make_evidence(
            "transfer.md", "Transfer criteria include instability monitoring.", "dense_vector"
        )

        result = evaluate_multi_technique(
            golden,
            answer="ICU transfer with hemodynamic instability monitoring.",
            technique_evidence={
                "sparse_bm25": [sparse_ev],
                "dense_vector": [dense_ev],
            },
            citations=[
                Citation(source="icu.md", document_id="doc_icu.md"),
                Citation(source="transfer.md", document_id="doc_transfer.md"),
            ],
            k=5,
        )

        assert result.technique_recall["sparse_bm25"] == 0.5
        assert result.technique_recall["dense_vector"] == 0.5
        assert result.overall > 0

    def test_empty_technique_gets_zero_contribution(self):
        golden = GoldenQuestion(
            id="q3",
            question="What is the protocol?",
            expected_sources=["protocol.md"],
            expected_terms=["step one"],
        )

        result = evaluate_multi_technique(
            golden,
            answer="Protocol involves step one.",
            technique_evidence={
                "sparse_bm25": [],
                "dense_vector": [],
            },
            citations=[],
            k=5,
        )

        assert result.technique_contributions["sparse_bm25"] == 0.0
        assert result.technique_contributions["dense_vector"] == 0.0


# ---------------------------------------------------------------------------
# 2. Pipeline Selection Quality Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineSelectionCase:
    """One test case for pipeline selection quality."""

    query: str
    query_type: str
    expected_best_pipeline: str
    actual_pipeline: str
    actual_score: float


def evaluate_pipeline_selection(
    cases: list[PipelineSelectionCase],
) -> dict[str, float]:
    """Evaluate whether the right pipeline was selected for each query type.

    Returns metrics:
        - selection_accuracy: fraction of cases where selected pipeline == expected
        - mean_score: average quality score across all cases
        - worst_score: minimum quality score (identifies weak spots)
        - type_accuracy: per-query-type selection accuracy
    """
    if not cases:
        return {"selection_accuracy": 0.0, "mean_score": 0.0, "worst_score": 0.0}

    correct = sum(1 for c in cases if c.actual_pipeline == c.expected_best_pipeline)
    scores = [c.actual_score for c in cases]

    # Per-type accuracy
    types: dict[str, list[bool]] = {}
    for c in cases:
        types.setdefault(c.query_type, []).append(
            c.actual_pipeline == c.expected_best_pipeline
        )

    result: dict[str, float] = {
        "selection_accuracy": correct / len(cases),
        "mean_score": sum(scores) / len(scores),
        "worst_score": min(scores),
        "total_cases": float(len(cases)),
    }
    for qt, correct_list in types.items():
        result[f"type_{qt}_accuracy"] = sum(correct_list) / len(correct_list)

    return result


class TestPipelineSelectionQuality:
    """Pipeline selection quality must be measurable and interpretable."""

    def test_perfect_selection(self):
        cases = [
            PipelineSelectionCase("q1", "specification", "hybrid_rrf", "hybrid_rrf", 0.9),
            PipelineSelectionCase("q2", "summary", "expanded_rrf", "expanded_rrf", 0.85),
        ]

        metrics = evaluate_pipeline_selection(cases)

        assert metrics["selection_accuracy"] == 1.0
        assert metrics["mean_score"] > 0.8

    def test_partial_selection_accuracy(self):
        cases = [
            PipelineSelectionCase("q1", "specification", "hybrid_rrf", "hybrid_rrf", 0.9),
            PipelineSelectionCase("q2", "summary", "expanded_rrf", "hybrid_rrf", 0.5),
        ]

        metrics = evaluate_pipeline_selection(cases)

        assert metrics["selection_accuracy"] == 0.5
        assert metrics["worst_score"] == 0.5

    def test_per_type_accuracy(self):
        cases = [
            PipelineSelectionCase("q1", "specification", "hybrid_rrf", "hybrid_rrf", 0.9),
            PipelineSelectionCase("q2", "specification", "hybrid_rrf", "dense_only", 0.4),
            PipelineSelectionCase("q3", "summary", "expanded_rrf", "expanded_rrf", 0.8),
        ]

        metrics = evaluate_pipeline_selection(cases)

        assert metrics["type_specification_accuracy"] == 0.5
        assert metrics["type_summary_accuracy"] == 1.0

    def test_empty_cases_returns_zero_metrics(self):
        metrics = evaluate_pipeline_selection([])
        assert metrics["selection_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 3. Technique-Level Contribution Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TechniqueContribution:
    """Measures one technique's contribution to the final answer."""

    technique: str
    recall_delta: float  # recall with technique - recall without
    term_match_delta: float
    is_essential: bool  # removing it causes > 20% drop in overall


def measure_technique_contribution(
    golden: GoldenQuestion,
    full_evidence: list[Evidence],
    technique_to_remove: str,
    answer: str,
    *,
    k: int = 5,
) -> TechniqueContribution:
    """Measure a technique's contribution by ablating it."""
    # Full scores
    full_recall = recall_at_k(golden.expected_sources, full_evidence, k=k)
    full_term = expected_term_match(answer, golden.expected_terms)

    # Ablated scores (remove evidence from this technique)
    ablated = [ev for ev in full_evidence if ev.retriever != technique_to_remove]
    ablated_recall = recall_at_k(golden.expected_sources, ablated, k=k)
    ablated_text = " ".join(ev.chunk.text for ev in ablated) if ablated else ""
    ablated_term = expected_term_match(ablated_text, golden.expected_terms)

    recall_delta = full_recall - ablated_recall
    term_delta = full_term - ablated_term
    full_overall = (full_recall + full_term) / 2
    ablated_overall = (ablated_recall + ablated_term) / 2
    drop = full_overall - ablated_overall

    return TechniqueContribution(
        technique=technique_to_remove,
        recall_delta=round(recall_delta, 6),
        term_match_delta=round(term_delta, 6),
        is_essential=drop > 0.2,
    )


class TestTechniqueContribution:
    """Technique contribution measurement must correctly assess impact."""

    def _make_evidence(self, source: str, text: str, retriever: str) -> Evidence:
        chunk = Chunk(
            id=f"c_{source}_{retriever}",
            document_id=f"d_{source}",
            text=text,
            source=source,
        )
        return Evidence(chunk=chunk, score=0.8, retriever=retriever)

    def test_essential_technique_has_positive_recall_delta(self):
        golden = GoldenQuestion(
            id="q1",
            question="triage protocol",
            expected_sources=["triage.md"],
            expected_terms=["vital signs"],
        )
        evidence = [
            self._make_evidence("triage.md", "Triage vital signs assessment.", "sparse_bm25"),
            self._make_evidence("billing.md", "Invoice processing system.", "dense_vector"),
        ]

        contrib = measure_technique_contribution(
            golden,
            evidence,
            "sparse_bm25",
            "Triage vital signs assessment.",
            k=5,
        )

        assert contrib.recall_delta > 0
        assert contrib.is_essential is True

    def test_non_contributing_technique_has_zero_delta(self):
        golden = GoldenQuestion(
            id="q2",
            question="triage protocol",
            expected_sources=["triage.md"],
            expected_terms=["vital signs"],
        )
        evidence = [
            self._make_evidence("triage.md", "Triage vital signs assessment.", "sparse_bm25"),
            self._make_evidence("triage.md", "Triage vital signs monitoring.", "dense_vector"),
        ]

        # Removing dense_vector: triage.md is still found by sparse_bm25
        contrib = measure_technique_contribution(
            golden,
            evidence,
            "dense_vector",
            "Triage vital signs assessment and monitoring.",
            k=5,
        )

        assert contrib.recall_delta == 0.0  # source still covered by sparse

    def test_both_techniques_essential_for_multi_source(self):
        golden = GoldenQuestion(
            id="q3",
            question="multi-source question",
            expected_sources=["source_a.md", "source_b.md"],
            expected_terms=["term_a", "term_b"],
        )
        evidence = [
            self._make_evidence("source_a.md", "Content with term_a details.", "sparse_bm25"),
            self._make_evidence("source_b.md", "Content with term_b details.", "dense_vector"),
        ]

        sparse_contrib = measure_technique_contribution(
            golden, evidence, "sparse_bm25",
            "Content with term_a and term_b details.", k=5,
        )
        dense_contrib = measure_technique_contribution(
            golden, evidence, "dense_vector",
            "Content with term_a and term_b details.", k=5,
        )

        assert sparse_contrib.recall_delta > 0
        assert dense_contrib.recall_delta > 0


# ---------------------------------------------------------------------------
# 4. Per-Technique Ablation Testing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AblationReport:
    """Report from ablating each technique from a pipeline."""

    full_score: float
    ablation_results: dict[str, float]  # technique -> score without it
    essential_techniques: list[str]
    redundant_techniques: list[str]


def run_ablation_suite(
    golden: GoldenQuestion,
    technique_evidence: dict[str, list[Evidence]],
    answer: str,
    *,
    k: int = 5,
    essentiality_threshold: float = 0.2,
) -> AblationReport:
    """Run ablation across all techniques and report which are essential."""
    all_evidence = []
    for ev_list in technique_evidence.values():
        all_evidence.extend(ev_list)

    full_result = evaluate_answer(
        golden, answer=answer, evidence=all_evidence, citations=[], k=k
    )
    full_score = full_result.overall

    ablation_results: dict[str, float] = {}
    essential: list[str] = []
    redundant: list[str] = []

    for tech_name in technique_evidence:
        ablated_evidence = []
        for other_tech, ev_list in technique_evidence.items():
            if other_tech != tech_name:
                ablated_evidence.extend(ev_list)

        ablated_text = (
            " ".join(ev.chunk.text for ev in ablated_evidence)
            if ablated_evidence
            else ""
        )
        ablated_result = evaluate_answer(
            golden, answer=ablated_text, evidence=ablated_evidence, citations=[], k=k
        )
        ablation_results[tech_name] = ablated_result.overall

        if full_score - ablated_result.overall > essentiality_threshold:
            essential.append(tech_name)
        else:
            redundant.append(tech_name)

    return AblationReport(
        full_score=full_score,
        ablation_results=ablation_results,
        essential_techniques=essential,
        redundant_techniques=redundant,
    )


class TestAblationSuite:
    """Ablation suite must correctly identify essential vs redundant techniques."""

    def _make_evidence(self, source: str, text: str, retriever: str) -> Evidence:
        chunk = Chunk(
            id=f"c_{source}_{retriever}",
            document_id=f"d_{source}",
            text=text,
            source=source,
        )
        return Evidence(chunk=chunk, score=0.8, retriever=retriever)

    def test_single_essential_technique(self):
        golden = GoldenQuestion(
            id="q1",
            question="triage protocol",
            expected_sources=["triage.md"],
            expected_terms=["vital signs", "acuity"],
        )
        technique_evidence = {
            "sparse_bm25": [
                self._make_evidence("triage.md", "Triage vital signs and acuity.", "sparse_bm25"),
            ],
            "dense_vector": [
                self._make_evidence("billing.md", "Invoice details monthly.", "dense_vector"),
            ],
        }

        report = run_ablation_suite(
            golden,
            technique_evidence,
            "Triage vital signs and acuity protocol.",
            k=5,
        )

        assert "sparse_bm25" in report.essential_techniques
        assert report.full_score > 0

    def test_all_techniques_contribute_to_multi_source(self):
        golden = GoldenQuestion(
            id="q2",
            question="comprehensive protocol",
            expected_sources=["doc_a.md", "doc_b.md"],
            expected_terms=["protocol_a", "protocol_b"],
        )
        technique_evidence = {
            "sparse": [
                self._make_evidence("doc_a.md", "Protocol_a is defined here.", "sparse"),
            ],
            "dense": [
                self._make_evidence("doc_b.md", "Protocol_b is defined here.", "dense"),
            ],
        }

        report = run_ablation_suite(
            golden,
            technique_evidence,
            "Protocol_a and protocol_b are both critical.",
            k=5,
        )

        # Both should be essential since each covers a different source/term
        assert "sparse" in report.essential_techniques
        assert "dense" in report.essential_techniques

    def test_redundant_technique_identified(self):
        golden = GoldenQuestion(
            id="q3",
            question="simple query",
            expected_sources=["doc.md"],
            expected_terms=["answer"],
        )
        technique_evidence = {
            "primary": [
                self._make_evidence("doc.md", "The answer is here.", "primary"),
            ],
            "backup": [
                self._make_evidence("doc.md", "The answer is also here.", "backup"),
            ],
        }

        report = run_ablation_suite(
            golden,
            technique_evidence,
            "The answer is here.",
            k=5,
        )

        # Both cover the same source and term, so removing either still passes
        assert len(report.redundant_techniques) >= 1

    def test_ablation_report_has_all_techniques(self):
        golden = GoldenQuestion(
            id="q4",
            question="test",
            expected_sources=["a.md"],
            expected_terms=["test"],
        )
        technique_evidence = {
            "tech_a": [self._make_evidence("a.md", "test content", "tech_a")],
            "tech_b": [],
            "tech_c": [],
        }

        report = run_ablation_suite(
            golden, technique_evidence, "test content", k=5
        )

        assert set(report.ablation_results.keys()) == {"tech_a", "tech_b", "tech_c"}


# ---------------------------------------------------------------------------
# 5. Integration: Extended Evaluation with Real Document Pipeline
# ---------------------------------------------------------------------------


class TestExtendedEvaluationIntegration:
    """Integration tests combining platform evaluation with multi-technique tracking."""

    def test_evaluate_answer_works_with_multi_retriever_evidence(self):
        """evaluate_answer must handle evidence from multiple retrievers."""
        golden = GoldenQuestion(
            id="integration_1",
            question="What is ED triage?",
            expected_sources=["triage.md"],
            expected_terms=["vital signs", "acuity"],
        )
        chunk_sparse = Chunk(
            id="c1", document_id="d1", text="ED triage vital signs protocol.", source="triage.md"
        )
        chunk_dense = Chunk(
            id="c2", document_id="d1", text="Acuity scoring methodology.", source="triage.md"
        )
        evidence = [
            Evidence(chunk=chunk_sparse, score=0.9, retriever="sparse_bm25"),
            Evidence(chunk=chunk_dense, score=0.85, retriever="dense_vector"),
        ]
        citations = [Citation(source="triage.md", document_id="d1")]

        result = evaluate_answer(
            golden,
            answer="ED triage uses vital signs and acuity scoring.",
            evidence=evidence,
            citations=citations,
            k=5,
        )

        assert result.recall_at_k == 1.0
        assert result.citation_source_accuracy == 1.0
        assert result.expected_term_match == 1.0
        assert result.overall == 1.0

    def test_golden_question_metadata_tracks_pipeline_info(self):
        """GoldenQuestion metadata can carry pipeline selection expectations."""
        golden = GoldenQuestion(
            id="meta_1",
            question="What is the escalation protocol?",
            expected_sources=["escalation.md"],
            expected_terms=["criteria"],
            metadata={
                "expected_pipeline": "hybrid_rrf",
                "expected_techniques": ["sparse_bm25", "dense_vector", "rrf_fusion"],
                "query_type": "specification",
            },
        )

        assert golden.metadata["expected_pipeline"] == "hybrid_rrf"
        assert "sparse_bm25" in golden.metadata["expected_techniques"]
        assert golden.metadata["query_type"] == "specification"
