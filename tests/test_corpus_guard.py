"""Tests for memory-bounded corpus guard (P7-04)."""

from __future__ import annotations

import logging
from pathlib import Path

from cam_rag.query import (
    _MAX_CORPUS_CHUNKS,
    query_document_folder,
)
from cam_rag.rag import RAGAppSpec


def test_max_corpus_chunks_constant():
    """The guard constant exists and is 50,000."""
    assert _MAX_CORPUS_CHUNKS == 50_000


def test_corpus_guard_truncates_and_warns(
    tmp_path: Path, caplog, monkeypatch
):
    """When chunks exceed the limit, a warning is logged and
    the chunk list is truncated to _MAX_CORPUS_CHUNKS.

    We temporarily lower the limit to make the test feasible
    without creating 50k+ real chunks.
    """
    # Lower the guard to a tiny value for testing
    monkeypatch.setattr(
        "cam_rag.query._MAX_CORPUS_CHUNKS", 3
    )

    # Create enough documents to produce > 3 chunks
    for i in range(6):
        (tmp_path / f"doc{i}.md").write_text(
            f"# Document {i}\n\nContent block {i} with "
            f"enough text to form a chunk.",
            encoding="utf-8",
        )

    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.WARNING, logger="cam_rag.query"
    ):
        answer = query_document_folder(
            tmp_path, "document content", spec
        )

    # The warning about truncation should have been emitted
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    truncation_warnings = [
        r
        for r in warnings
        if "corpus exceeds maximum chunk limit" in r.message
    ]
    assert len(truncation_warnings) >= 1

    # The pipeline still produces a valid answer
    assert answer.answer
    assert answer.trace.stages


def test_corpus_guard_no_warning_under_limit(
    tmp_path: Path, caplog, monkeypatch
):
    """No truncation warning when chunks are under the limit."""
    monkeypatch.setattr(
        "cam_rag.query._MAX_CORPUS_CHUNKS", 1000
    )

    (tmp_path / "small.md").write_text(
        "# Small\nA brief document.", encoding="utf-8"
    )

    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.WARNING, logger="cam_rag.query"
    ):
        answer = query_document_folder(
            tmp_path, "brief document", spec
        )

    truncation_warnings = [
        r
        for r in caplog.records
        if "corpus exceeds maximum chunk limit" in r.message
    ]
    assert truncation_warnings == []
    assert answer.answer


def test_corpus_guard_pipeline_still_works_after_truncation(
    tmp_path: Path, monkeypatch, caplog
):
    """After truncation the pipeline produces valid evidence."""
    monkeypatch.setattr(
        "cam_rag.query._MAX_CORPUS_CHUNKS", 2
    )

    for i in range(5):
        (tmp_path / f"doc{i}.md").write_text(
            f"# Protocol {i}\n\nClinical protocol {i} "
            f"for emergency department triage.",
            encoding="utf-8",
        )

    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.WARNING, logger="cam_rag.query"
    ):
        answer = query_document_folder(
            tmp_path, "clinical protocol triage", spec
        )

    # Still returns a valid answer with evidence
    assert answer.answer
    # Evidence count is bounded by the truncated corpus
    assert answer.trace.retrieval_stats["evidence"] <= 2
    # Truncation warning was logged
    truncation_warnings = [
        r
        for r in caplog.records
        if "corpus exceeds maximum chunk limit" in r.message
    ]
    assert len(truncation_warnings) >= 1
