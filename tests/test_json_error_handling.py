"""Tests for R6-02: JSON error handling in folder.py."""

from __future__ import annotations

import warnings
from pathlib import Path

from cam_rag.documents.folder import _read_json_documents


def test_malformed_jsonl_line_skipped_with_warning(tmp_path: Path):
    """Malformed JSONL lines are skipped; valid lines still parse."""
    root = tmp_path
    jsonl_file = root / "data.jsonl"
    jsonl_file.write_text(
        '{"text": "first line is good"}\n'
        "this is not json\n"
        '{"text": "third line is good"}\n',
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, jsonl_file)

    assert len(docs) == 2
    assert docs[0].text == "first line is good"
    assert docs[1].text == "third line is good"
    assert len(caught) == 1
    assert "malformed JSONL line 2" in str(caught[0].message)


def test_all_jsonl_lines_malformed_returns_empty(tmp_path: Path):
    """If every JSONL line is malformed, an empty list is returned."""
    root = tmp_path
    jsonl_file = root / "bad.jsonl"
    jsonl_file.write_text("not json\nalso not json\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, jsonl_file)

    assert docs == []
    assert len(caught) == 2


def test_malformed_json_file_returns_empty_with_warning(tmp_path: Path):
    """A .json file with invalid JSON returns an empty list and warns."""
    root = tmp_path
    json_file = root / "bad.json"
    json_file.write_text("{not valid json", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, json_file)

    assert docs == []
    assert len(caught) == 1
    assert "could not parse JSON file" in str(caught[0].message)


def test_valid_json_file_still_works(tmp_path: Path):
    """Ensure valid JSON files parse correctly after adding error handling."""
    root = tmp_path
    json_file = root / "good.json"
    json_file.write_text(
        '[{"text": "alpha"}, {"text": "beta"}]',
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, json_file)

    assert len(docs) == 2
    assert docs[0].text == "alpha"
    assert docs[1].text == "beta"
    assert len(caught) == 0


def test_valid_jsonl_file_still_works(tmp_path: Path):
    """Ensure valid JSONL files parse correctly after adding error handling."""
    root = tmp_path
    jsonl_file = root / "good.jsonl"
    jsonl_file.write_text(
        '{"text": "line one"}\n{"text": "line two"}\n',
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, jsonl_file)

    assert len(docs) == 2
    assert len(caught) == 0


def test_unreadable_file_returns_empty_with_warning(tmp_path: Path):
    """An unreadable file returns empty list with an OSError warning."""
    root = tmp_path
    json_file = root / "locked.json"
    json_file.write_text('{"text": "hello"}', encoding="utf-8")
    # Remove read permission
    json_file.chmod(0o000)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            docs = _read_json_documents(root, json_file)

        assert docs == []
        assert len(caught) == 1
        assert "could not read JSON file" in str(caught[0].message)
    finally:
        # Restore permissions so tmp_path cleanup can remove the file
        json_file.chmod(0o644)


def test_empty_jsonl_returns_empty(tmp_path: Path):
    """An empty JSONL file returns an empty list with no warnings."""
    root = tmp_path
    jsonl_file = root / "empty.jsonl"
    jsonl_file.write_text("", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, jsonl_file)

    assert docs == []
    assert len(caught) == 0


def test_mixed_valid_invalid_jsonl(tmp_path: Path):
    """A JSONL file mixing valid, invalid, and empty lines."""
    root = tmp_path
    jsonl_file = root / "mixed.jsonl"
    jsonl_file.write_text(
        '{"text": "good1"}\n'
        "\n"
        "bad line\n"
        '{"text": "good2"}\n'
        "{truncated\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        docs = _read_json_documents(root, jsonl_file)

    assert len(docs) == 2
    assert docs[0].text == "good1"
    assert docs[1].text == "good2"
    # Two bad lines: line 3 and line 5
    assert len(caught) == 2
