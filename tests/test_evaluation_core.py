from cam_rag.evaluation import (
    GoldenQuestion,
    citation_source_accuracy,
    evaluate_answer,
    expected_term_match,
    load_golden_questions_jsonl,
    no_evidence_behavior,
    recall_at_k,
    save_golden_questions_jsonl,
)
from cam_rag.rag.models import Chunk, Citation, Evidence


def test_recall_at_k_matches_expected_sources_from_evidence():
    evidence = [
        Evidence(
            chunk=Chunk(id="c1", document_id="d1", text="wrong", source="other.md"),
            score=1.0,
            retriever="test",
        ),
        Evidence(
            chunk=Chunk(id="c2", document_id="d2", text="triage", source="triage.md"),
            score=0.9,
            retriever="test",
        ),
    ]

    assert recall_at_k(["triage.md", "policy.md"], evidence, k=2) == 0.5
    assert recall_at_k(["triage.md"], evidence, k=1) == 0.0


def test_citation_source_accuracy_counts_only_expected_citations():
    citations = [
        Citation(source="triage.md", document_id="d1"),
        Citation(source="other.md", document_id="d2"),
    ]

    assert citation_source_accuracy(["triage.md"], citations) == 0.5
    assert citation_source_accuracy(["triage.md"], []) == 0.0


def test_expected_term_match_is_case_insensitive_and_phrase_based():
    answer = "Use ED triage vital signs before assigning the acuity score."

    assert expected_term_match(answer, ["ed triage", "acuity score", "missing"]) == 0.666667
    assert expected_term_match(answer, []) == 1.0


def test_no_evidence_behavior_requires_refusal_without_support():
    assert (
        no_evidence_behavior(
            expected_no_evidence=True,
            answer="I cannot answer because there is no evidence in the corpus.",
            evidence=[],
            citations=[],
        )
        == 1.0
    )
    assert (
        no_evidence_behavior(
            expected_no_evidence=True,
            answer="The answer is definitely yes.",
            evidence=[],
            citations=[],
        )
        == 0.0
    )
    assert (
        no_evidence_behavior(
            expected_no_evidence=True,
            answer="No evidence.",
            evidence=["source.md"],
            citations=[],
        )
        == 0.0
    )


def test_evaluate_answer_returns_result_with_overall_and_metrics():
    golden = GoldenQuestion(
        id="q1",
        question="How is triage scored?",
        expected_sources=["triage.md"],
        expected_terms=["vital signs"],
    )
    evidence = [
        {"source": "triage.md", "text": "vital signs"},
    ]
    citations = [
        {"source": "triage.md"},
    ]

    result = evaluate_answer(
        golden,
        answer="Triage scoring uses vital signs.",
        evidence=evidence,
        citations=citations,
        k=1,
    )

    assert result.question_id == "q1"
    assert result.recall_at_k == 1.0
    assert result.citation_source_accuracy == 1.0
    assert result.expected_term_match == 1.0
    assert result.no_evidence_behavior == 1.0
    assert result.overall == 1.0
    assert result.to_dict()["metrics"]["recall_at_k"] == 1.0


def test_golden_questions_jsonl_round_trip(tmp_path):
    path = tmp_path / "golden.jsonl"
    questions = [
        GoldenQuestion(
            id="q1",
            question="What is supported?",
            expected_sources=["source.md"],
            expected_terms=["supported"],
            metadata={"split": "smoke"},
        ),
        GoldenQuestion(id="q2", question="Unknown?", no_evidence=True),
    ]

    save_golden_questions_jsonl(path, questions)

    assert load_golden_questions_jsonl(path) == questions
