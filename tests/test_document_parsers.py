import builtins
from pathlib import Path

import pytest

from cam_rag.documents.parsers import (
    MissingParserDependencyError,
    extract_title,
    parse_docx_text,
    parse_pdf_text,
)


def test_extract_title_prefers_markdown_heading():
    assert extract_title("Intro\n# Trauma Protocol\nBody", "fallback") == "Trauma Protocol"
    assert extract_title("Plain title\nBody", "fallback") == "Plain title"
    assert extract_title("", "fallback") == "fallback"


def test_parse_pdf_missing_dependency_error(monkeypatch, tmp_path: Path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"pymupdf", "pymupdf4llm"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(MissingParserDependencyError, match="PDF ingestion requires"):
        parse_pdf_text(pdf)


def test_parse_docx_missing_dependency_error(monkeypatch, tmp_path: Path):
    docx = tmp_path / "sample.docx"
    docx.write_bytes(b"not really a docx")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(MissingParserDependencyError, match="DOCX ingestion requires"):
        parse_docx_text(docx)
