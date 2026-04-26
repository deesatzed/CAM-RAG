"""Evaluation helpers for golden-question RAG benchmarks."""

from cam_rag.evaluation.core import (
    EvaluationResult,
    GoldenQuestion,
    citation_source_accuracy,
    evaluate_answer,
    expected_term_match,
    load_golden_questions_jsonl,
    no_evidence_behavior,
    recall_at_k,
    save_evaluation_results_jsonl,
    save_golden_questions_jsonl,
)

__all__ = [
    "EvaluationResult",
    "GoldenQuestion",
    "citation_source_accuracy",
    "evaluate_answer",
    "expected_term_match",
    "load_golden_questions_jsonl",
    "no_evidence_behavior",
    "recall_at_k",
    "save_evaluation_results_jsonl",
    "save_golden_questions_jsonl",
]
