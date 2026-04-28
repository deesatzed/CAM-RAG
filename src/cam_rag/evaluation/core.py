"""Dependency-free evaluation primitives for RAG answers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """Expected behavior for one benchmark question."""

    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)
    no_evidence: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuestion:
        no_evidence = bool(data.get("no_evidence", False))
        if "should_answer" in data:
            no_evidence = not bool(data["should_answer"])
        return cls(
            id=str(data.get("id", "")),
            question=str(data["question"]),
            expected_sources=[str(item) for item in data.get("expected_sources", [])],
            expected_terms=[
                str(item)
                for item in data.get("expected_terms", data.get("must_include", []))
            ],
            no_evidence=no_evidence,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metric summary for one evaluated answer."""

    question_id: str
    recall_at_k: float
    citation_source_accuracy: float
    expected_term_match: float
    no_evidence_behavior: float
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        scores = [
            self.recall_at_k,
            self.citation_source_accuracy,
            self.expected_term_match,
            self.no_evidence_behavior,
        ]
        return round(sum(scores) / len(scores), 6)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overall"] = self.overall
        return data


def load_golden_questions_jsonl(path: str | Path) -> list[GoldenQuestion]:
    """Load golden questions from a JSONL file."""

    questions: list[GoldenQuestion] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid golden question at line {line_number}: expected object")
            question = GoldenQuestion.from_dict(payload)
            if not question.id:
                question = GoldenQuestion(
                    id=f"q{line_number}",
                    question=question.question,
                    expected_sources=question.expected_sources,
                    expected_terms=question.expected_terms,
                    no_evidence=question.no_evidence,
                    metadata=question.metadata,
                )
            questions.append(question)
    return questions


def save_golden_questions_jsonl(path: str | Path, questions: Iterable[GoldenQuestion]) -> None:
    """Save golden questions to JSONL."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), sort_keys=True) + "\n")


def save_evaluation_results_jsonl(path: str | Path, results: Iterable[EvaluationResult]) -> None:
    """Save evaluation results to JSONL."""

    with Path(path).open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def recall_at_k(expected_sources: Iterable[str], retrieved: Iterable[Any], k: int) -> float:
    """Return source recall in the first k retrieved evidence items."""

    expected = _normalized_set(expected_sources)
    if not expected:
        return 1.0
    if k <= 0:
        return 0.0

    retrieved_sources = {_normalize_source(_source_from_item(item)) for item in list(retrieved)[:k]}
    retrieved_sources.discard("")
    return round(len(expected & retrieved_sources) / len(expected), 6)


def citation_source_accuracy(expected_sources: Iterable[str], citations: Iterable[Any]) -> float:
    """Return the share of citations whose source is expected."""

    citation_sources = [_normalize_source(_source_from_item(item)) for item in citations]
    citation_sources = [source for source in citation_sources if source]
    if not citation_sources:
        return 1.0 if not _normalized_set(expected_sources) else 0.0

    expected = _normalized_set(expected_sources)
    if not expected:
        return 0.0
    correct = sum(1 for source in citation_sources if source in expected)
    return round(correct / len(citation_sources), 6)


def expected_term_match(answer: str, expected_terms: Iterable[str]) -> float:
    """Return the share of expected terms found in the answer text."""

    terms = [term for term in expected_terms if term.strip()]
    if not terms:
        return 1.0
    normalized_answer = _normalize_text(answer)
    matched = sum(1 for term in terms if _normalize_text(term) in normalized_answer)
    return round(matched / len(terms), 6)


def no_evidence_behavior(
    *,
    expected_no_evidence: bool,
    answer: str,
    evidence: Iterable[Any] = (),
    citations: Iterable[Any] = (),
) -> float:
    """Score whether an answer behaves correctly when no supporting evidence is expected."""

    evidence_items = list(evidence)
    citation_items = list(citations)
    if not expected_no_evidence:
        return 1.0
    if evidence_items or citation_items:
        return 0.0
    return 1.0 if _looks_like_no_evidence_answer(answer) else 0.0


def evaluate_answer(
    golden: GoldenQuestion,
    *,
    answer: str,
    evidence: Iterable[Any] = (),
    citations: Iterable[Any] = (),
    k: int = 5,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate one answer against a golden question."""

    evidence_items = list(evidence)
    citation_items = list(citations)
    metrics = {
        "recall_at_k": recall_at_k(golden.expected_sources, evidence_items, k),
        "citation_source_accuracy": citation_source_accuracy(
            golden.expected_sources,
            citation_items,
        ),
        "expected_term_match": expected_term_match(answer, golden.expected_terms),
        "no_evidence_behavior": no_evidence_behavior(
            expected_no_evidence=golden.no_evidence,
            answer=answer,
            evidence=evidence_items,
            citations=citation_items,
        ),
    }
    return EvaluationResult(
        question_id=golden.id,
        recall_at_k=metrics["recall_at_k"],
        citation_source_accuracy=metrics["citation_source_accuracy"],
        expected_term_match=metrics["expected_term_match"],
        no_evidence_behavior=metrics["no_evidence_behavior"],
        metrics=metrics,
        metadata={"expected_no_evidence": golden.no_evidence, **(metadata or {})},
    )


def _source_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    source = _field(item, "source")
    if source:
        return str(source)
    chunk = _field(item, "chunk")
    if chunk is not None:
        chunk_source = _field(chunk, "source")
        if chunk_source:
            return str(chunk_source)
    metadata = _field(item, "metadata")
    if isinstance(metadata, dict):
        for key in ("source", "document_id", "doc_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    for key in ("document_id", "doc_id", "id"):
        value = _field(item, key)
        if value:
            return str(value)
    return ""


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    if is_dataclass(item) or hasattr(item, name):
        return getattr(item, name, None)
    return None


def _normalized_set(values: Iterable[str]) -> set[str]:
    return {_normalize_source(value) for value in values if _normalize_source(value)}


def _normalize_source(value: Any) -> str:
    return str(value).strip().casefold()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _looks_like_no_evidence_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    if not normalized:
        return True
    markers = (
        "no evidence",
        "no cited evidence",
        "not enough evidence",
        "insufficient evidence",
        "cannot answer",
        "can't answer",
        "do not know",
        "don't know",
        "not found",
        "no supporting",
    )
    return any(marker in normalized for marker in markers)
