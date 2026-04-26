from pathlib import Path

from cam_rag.evaluate import (
    evaluate_document_folder,
    evaluate_document_folder_jsonl,
    load_golden_jsonl,
)
from cam_rag.evaluation import GoldenQuestion
from cam_rag.rag import RAGAppSpec


def test_load_golden_jsonl_assigns_missing_ids(tmp_path: Path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"question":"What is triage?","expected_sources":["triage.md"]}\n',
        encoding="utf-8",
    )

    questions = load_golden_jsonl(golden)

    assert questions[0].id == "q1"
    assert questions[0].expected_sources == ["triage.md"]


def test_evaluate_document_folder_scores_sources_terms_and_no_evidence(tmp_path: Path):
    (tmp_path / "triage.md").write_text(
        "# Triage\n\nED triage requires vital signs.",
        encoding="utf-8",
    )
    questions = [
        GoldenQuestion(
            id="triage",
            question="ED triage vital signs",
            expected_sources=["triage.md"],
            expected_terms=["vital", "signs"],
        ),
        GoldenQuestion(
            id="missing",
            question="airway protocol",
            no_evidence=True,
        ),
    ]

    report = evaluate_document_folder(
        tmp_path,
        questions,
        RAGAppSpec(name="eval", supported_extensions=(".md",)),
    )

    assert report.metrics["total"] == 2
    assert report.metrics["recall_at_k"] == 1.0
    assert report.metrics["citation_source_accuracy"] == 1.0
    assert report.metrics["expected_term_match"] == 1.0
    assert report.metrics["no_evidence_accuracy"] == 1.0


def test_evaluate_document_folder_jsonl(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "icu.md").write_text(
        "# ICU\n\nICU transfer criteria include instability.",
        encoding="utf-8",
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id":"icu","question":"ICU transfer criteria","expected_sources":["icu.md"],'
        '"expected_terms":["criteria"]}\n',
        encoding="utf-8",
    )

    report = evaluate_document_folder_jsonl(
        docs,
        golden,
        RAGAppSpec(name="eval", supported_extensions=(".md",)),
    )

    assert report.cases[0].recall_at_k == 1.0
    assert report.to_dict()["metrics"]["total"] == 1
