from pathlib import Path

from cam_rag.query import query_document_folder
from cam_rag.rag import RAGAppSpec


def test_query_document_folder_returns_cited_evidence(tmp_path: Path):
    (tmp_path / "triage.md").write_text(
        "# Triage\n\nED triage requires immediate vital signs and acuity assignment.",
        encoding="utf-8",
    )
    (tmp_path / "billing.md").write_text(
        "# Billing\n\nInvoices are reviewed monthly.",
        encoding="utf-8",
    )

    answer = query_document_folder(
        tmp_path,
        "ED triage vital signs",
        RAGAppSpec(name="docs", supported_extensions=(".md",)),
    )

    assert answer.grounded is True
    assert answer.citations
    assert answer.citations[0].source == "triage.md"
    assert answer.evidence[0].retriever == "sparse_bm25"
    assert "vital" in answer.evidence[0].signals["matched_terms"]


def test_query_document_folder_reports_no_evidence(tmp_path: Path):
    (tmp_path / "billing.md").write_text(
        "# Billing\n\nInvoices are reviewed monthly.",
        encoding="utf-8",
    )

    answer = query_document_folder(
        tmp_path,
        "trauma airway protocol",
        RAGAppSpec(name="docs", supported_extensions=(".md",)),
    )

    assert answer.grounded is False
    assert answer.citations == []
    assert "No cited evidence" in answer.answer
