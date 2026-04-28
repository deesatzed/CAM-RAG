"""Tests for structured logging configuration (S5-09)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cam_rag.logging_config import (
    StructuredLogger,
    configure_logging,
)
from cam_rag.query import query_document_folder
from cam_rag.rag import RAGAppSpec

# -- configure_logging ---------------------------------------------------


def test_configure_logging_sets_level_info():
    configure_logging(level="WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    # Reset to avoid side-effects on other tests
    configure_logging(level="INFO")


def test_configure_logging_sets_level_debug():
    configure_logging(level="DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    configure_logging(level="INFO")


# -- JSON format ---------------------------------------------------------


def test_json_format_produces_valid_json(capfd):
    configure_logging(level="DEBUG", json_format=True)
    test_logger = logging.getLogger("test.json_format")
    test_logger.info("hello json")

    captured = capfd.readouterr()
    # JSON output goes to stderr
    line = captured.err.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["message"] == "hello json"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.json_format"
    assert "timestamp" in parsed

    # Reset
    configure_logging(level="INFO", json_format=False)


# -- Human format --------------------------------------------------------


def test_human_format_is_readable(capfd):
    configure_logging(level="DEBUG", json_format=False)
    test_logger = logging.getLogger("test.human_format")
    test_logger.info("hello human")

    captured = capfd.readouterr()
    output = captured.err.strip()
    assert "hello human" in output
    assert "INFO" in output
    assert "test.human_format" in output

    configure_logging(level="INFO", json_format=False)


# -- StructuredLogger extra fields ---------------------------------------


def test_structured_logger_adds_extras_in_json_mode(capfd):
    configure_logging(level="DEBUG", json_format=True)
    slog = StructuredLogger("test.structured")
    slog.info(
        "query received",
        query_type="synthesis",
        evidence_count=5,
    )

    captured = capfd.readouterr()
    line = captured.err.strip().splitlines()[-1]
    parsed = json.loads(line)
    assert parsed["query_type"] == "synthesis"
    assert parsed["evidence_count"] == 5
    assert "query received" in parsed["message"]

    configure_logging(level="INFO", json_format=False)


def test_structured_logger_adds_extras_in_human_mode(
    capfd,
):
    configure_logging(level="DEBUG", json_format=False)
    slog = StructuredLogger("test.structured_human")
    slog.info("indexed", doc_count=42)

    captured = capfd.readouterr()
    output = captured.err.strip()
    assert "indexed" in output
    assert "doc_count=42" in output

    configure_logging(level="INFO", json_format=False)


# -- query.py emits expected log messages --------------------------------


def test_query_py_logs_query_received(
    tmp_path: Path, caplog
):
    (tmp_path / "triage.md").write_text(
        "# Triage\nED triage vital signs check.",
        encoding="utf-8",
    )
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.INFO, logger="cam_rag.query"
    ):
        query_document_folder(
            tmp_path, "ED triage vital signs", spec
        )

    messages = [r.message for r in caplog.records]
    received = [m for m in messages if "query received" in m]
    assert len(received) >= 1
    assert any("summary" in m for m in received)


def test_query_py_logs_evidence_count(
    tmp_path: Path, caplog
):
    (tmp_path / "triage.md").write_text(
        "# Triage\nED triage vital signs check.",
        encoding="utf-8",
    )
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.INFO, logger="cam_rag.query"
    ):
        query_document_folder(
            tmp_path, "ED triage vital signs", spec
        )

    messages = [r.message for r in caplog.records]
    assert any("evidence_count=" in m for m in messages)


def test_query_py_logs_confidence(
    tmp_path: Path, caplog
):
    (tmp_path / "triage.md").write_text(
        "# Triage\nED triage vital signs check.",
        encoding="utf-8",
    )
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.INFO, logger="cam_rag.query"
    ):
        query_document_folder(
            tmp_path, "ED triage vital signs", spec
        )

    messages = [r.message for r in caplog.records]
    assert any("confidence" in m for m in messages)


def test_query_py_warns_no_evidence(
    tmp_path: Path, caplog
):
    (tmp_path / "billing.md").write_text(
        "# Billing\nInvoices monthly.", encoding="utf-8"
    )
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.WARNING, logger="cam_rag.query"
    ):
        query_document_folder(
            tmp_path, "trauma airway protocol", spec
        )

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    warning_msgs = [r.message for r in warnings]
    assert any(
        "no evidence found" in m for m in warning_msgs
    )


def test_query_py_warns_low_confidence(
    tmp_path: Path, caplog
):
    # A single short doc yields low confidence
    (tmp_path / "notes.md").write_text(
        "# Notes\nBrief note.", encoding="utf-8"
    )
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with caplog.at_level(
        logging.WARNING, logger="cam_rag.query"
    ):
        answer = query_document_folder(
            tmp_path, "brief note", spec
        )

    if answer.confidence < 0.4:
        warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        warning_msgs = [r.message for r in warnings]
        assert any(
            "low confidence" in m for m in warning_msgs
        )
