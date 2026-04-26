"""End-to-end document-folder evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cam_rag.evaluation import EvaluationResult, GoldenQuestion, evaluate_answer
from cam_rag.evaluation import load_golden_questions_jsonl as load_golden_jsonl
from cam_rag.query import query_document_folder
from cam_rag.rag.spec import RAGAppSpec


@dataclass(frozen=True, slots=True)
class FolderEvaluationReport:
    """Aggregate document-folder evaluation report."""

    results: list[EvaluationResult]
    metrics: dict[str, float | int]

    @property
    def cases(self) -> list[EvaluationResult]:
        """Backward-compatible alias for per-question results."""

        return self.results

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "cases": [result.to_dict() for result in self.results],
        }


def evaluate_document_folder(
    docs_dir: str | Path,
    golden_questions: list[GoldenQuestion],
    spec: RAGAppSpec,
) -> FolderEvaluationReport:
    """Evaluate a document folder against golden retrieval/citation expectations."""

    results: list[EvaluationResult] = []
    for golden in golden_questions:
        answer = query_document_folder(docs_dir, golden.question, spec)
        answer_text = " ".join(
            [answer.answer, *[item.chunk.text for item in answer.evidence]]
        )
        result = evaluate_answer(
            golden,
            answer=answer_text,
            evidence=answer.evidence,
            citations=answer.citations,
            k=spec.retrieval_top_k,
            metadata={
                "confidence": answer.confidence,
                "grounded": answer.grounded,
                "cited_sources": [citation.source for citation in answer.citations],
            },
        )
        results.append(result)

    return FolderEvaluationReport(results=results, metrics=_aggregate(results))


def evaluate_document_folder_jsonl(
    docs_dir: str | Path,
    golden_jsonl: str | Path,
    spec: RAGAppSpec,
) -> FolderEvaluationReport:
    """Evaluate a folder against a JSONL golden set."""

    return evaluate_document_folder(docs_dir, load_golden_jsonl(golden_jsonl), spec)


def _aggregate(results: list[EvaluationResult]) -> dict[str, float | int]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "recall_at_k": 0.0,
            "citation_source_accuracy": 0.0,
            "expected_term_match": 0.0,
            "no_evidence_accuracy": 0.0,
            "overall": 0.0,
            "mean_confidence": 0.0,
        }

    return {
        "total": total,
        "recall_at_k": _mean(result.recall_at_k for result in results),
        "citation_source_accuracy": _mean(
            result.citation_source_accuracy for result in results
        ),
        "expected_term_match": _mean(result.expected_term_match for result in results),
        "no_evidence_accuracy": _mean(
            result.no_evidence_behavior
            for result in results
            if result.metadata.get("expected_no_evidence")
        ),
        "overall": _mean(result.overall for result in results),
        "mean_confidence": _mean(
            float(result.metadata.get("confidence", 0.0)) for result in results
        ),
    }


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
