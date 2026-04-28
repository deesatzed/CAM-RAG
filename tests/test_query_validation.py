"""Tests for query input validation (S5-03)."""

from __future__ import annotations

import pytest

from cam_rag.query import _MAX_QUERY_LENGTH, _validate_query, query
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec


class TestValidateQuery:
    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_query("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_query("   \t\n  ")

    def test_non_string_rejected(self):
        with pytest.raises(ValueError, match="string"):
            _validate_query(42)  # type: ignore[arg-type]

    def test_over_length_rejected(self):
        with pytest.raises(ValueError, match="maximum length"):
            _validate_query("x" * (_MAX_QUERY_LENGTH + 1))

    def test_at_limit_accepted(self):
        result = _validate_query("x" * _MAX_QUERY_LENGTH)
        assert len(result) == _MAX_QUERY_LENGTH

    def test_normal_query_passes(self):
        result = _validate_query("  What is retrieval augmented generation?  ")
        assert result == "What is retrieval augmented generation?"

    def test_whitespace_stripped(self):
        assert _validate_query("  hello  ") == "hello"


class TestQueryFunctionValidation:
    def test_query_rejects_empty(self):
        spec = RAGAppSpec(name="test")
        doc = CorpusDocument(
            id="d1", text="Some document text.", source="t.md", title="T"
        )
        with pytest.raises(ValueError, match="empty"):
            query("", [doc], spec)

    def test_query_rejects_non_string(self):
        spec = RAGAppSpec(name="test")
        doc = CorpusDocument(
            id="d1", text="Some document text.", source="t.md", title="T"
        )
        with pytest.raises(ValueError, match="string"):
            query(123, [doc], spec)  # type: ignore[arg-type]


class TestOutputTruncation:
    def test_long_query_truncated_in_answer(self):
        spec = RAGAppSpec(name="test")
        doc = CorpusDocument(
            id="d1",
            text="The quick brown fox jumps over the lazy dog.",
            source="t.md",
            title="T",
        )
        long_q = "a" * 300
        result = query(long_q, [doc], spec)
        # The echoed query in the answer should be truncated to 200 + "..."
        assert "..." in result.answer
        assert long_q not in result.answer
