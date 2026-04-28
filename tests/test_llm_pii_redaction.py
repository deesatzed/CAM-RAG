"""Tests for LLM-backed PII/PHI detection (Phase 2).

These tests exercise the ollama-backed NER model. Tests that hit the
live model are marked with ``pytest.mark.llm_live`` so CI can skip them
when ollama is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cam_rag.query import query_document_folder
from cam_rag.rag.spec import RAGAppSpec, RAGPolicy
from cam_rag.verification.sensitive import (
    _find_entity_position,
    _parse_llm_entities,
    redact_text,
    scan_text,
)

# ---------------------------------------------------------------------------
# Unit tests for LLM response parsing (no model required)
# ---------------------------------------------------------------------------


class TestParseLlmEntities:
    def test_valid_json_array(self):
        text = "Patient John Smith was admitted."
        raw = json.dumps([
            {"type": "NAME", "value": "John Smith", "start": 8, "end": 18},
        ])
        matches = _parse_llm_entities(text, raw)
        assert len(matches) == 1
        assert matches[0].pattern_name == "llm_name"
        assert matches[0].type == "phi"
        assert text[matches[0].start:matches[0].end] == "John Smith"

    def test_empty_response(self):
        matches = _parse_llm_entities("hello", "")
        assert matches == []

    def test_no_json_array(self):
        matches = _parse_llm_entities("hello", "No entities found.")
        assert matches == []

    def test_malformed_json(self):
        matches = _parse_llm_entities("hello", "[{broken json")
        assert matches == []

    def test_position_correction(self):
        """LLM offsets are approximate; parser should find actual position."""
        text = "Call Dr. Sarah Johnson at the clinic."
        raw = json.dumps([
            {"type": "NAME", "value": "Sarah Johnson", "start": 5, "end": 20},
        ])
        matches = _parse_llm_entities(text, raw)
        assert len(matches) == 1
        actual_start = text.index("Sarah Johnson")
        assert matches[0].start == actual_start
        assert matches[0].end == actual_start + len("Sarah Johnson")

    def test_value_not_in_text_skipped(self):
        """If the LLM hallucinates a value not in the text, skip it."""
        text = "The patient was discharged."
        raw = json.dumps([
            {"type": "NAME", "value": "Nonexistent Person", "start": 0, "end": 18},
        ])
        matches = _parse_llm_entities(text, raw)
        assert matches == []

    def test_pii_entity_types(self):
        text = "Address: 123 Oak St, SSN: 111-22-3333"
        raw = json.dumps([
            {"type": "ADDRESS", "value": "123 Oak St", "start": 9, "end": 19},
            {"type": "SSN", "value": "111-22-3333", "start": 26, "end": 37},
        ])
        matches = _parse_llm_entities(text, raw)
        address_m = [m for m in matches if m.pattern_name == "llm_address"]
        ssn_m = [m for m in matches if m.pattern_name == "llm_ssn"]
        assert len(address_m) == 1
        assert address_m[0].type == "pii"
        assert len(ssn_m) == 1
        assert ssn_m[0].type == "pii"

    def test_json_embedded_in_text(self):
        """LLM may return text before/after the JSON array."""
        text = "Age 45 patient."
        raw = 'Here are the entities:\n[{"type":"AGE","value":"45","start":4,"end":6}]\nDone.'
        matches = _parse_llm_entities(text, raw)
        assert len(matches) == 1
        assert matches[0].pattern_name == "llm_age"

    def test_multiple_entities(self):
        text = "John Smith, age 67, MRN: 1234567"
        raw = json.dumps([
            {"type": "NAME", "value": "John Smith", "start": 0, "end": 10},
            {"type": "AGE", "value": "67", "start": 16, "end": 18},
            {"type": "MRN", "value": "1234567", "start": 25, "end": 32},
        ])
        matches = _parse_llm_entities(text, raw)
        assert len(matches) == 3


class TestFindEntityPosition:
    def test_exact_match(self):
        assert _find_entity_position("hello world", "world", {"start": 6}) == 6

    def test_offset_correction(self):
        # LLM says start=5, actual is 6
        assert _find_entity_position("hello world", "world", {"start": 5}) == 6

    def test_fallback_full_search(self):
        assert _find_entity_position("hello world", "world", {"start": -1}) == 6

    def test_not_found(self):
        assert _find_entity_position("hello world", "mars", {"start": 0}) == -1


# ---------------------------------------------------------------------------
# Integration: scan_text with LLM (mocked ollama)
# ---------------------------------------------------------------------------


class TestScanTextWithLlm:
    def _mock_ollama_response(self, entities: list[dict]) -> dict:
        return {"response": json.dumps(entities)}

    def test_llm_adds_entities_regex_missed(self):
        """LLM detects names/ages that regex cannot."""
        text = "Dr. Sarah Johnson treated a 45-year-old in Boston."

        llm_entities = [
            {"type": "NAME", "value": "Sarah Johnson", "start": 4, "end": 17},
            {"type": "AGE", "value": "45", "start": 28, "end": 30},
            {"type": "ADDRESS", "value": "Boston", "start": 44, "end": 50},
        ]

        with patch(
            "cam_rag.verification.sensitive.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = json.dumps(
                self._mock_ollama_response(llm_entities)
            ).encode()

            matches = scan_text(
                text, check_pii=True, check_phi=True, use_llm=True,
            )

        # Regex finds nothing in this text; LLM should find all 3
        llm_matches = [m for m in matches if m.pattern_name.startswith("llm_")]
        assert len(llm_matches) == 3

    def test_llm_deduplicates_with_regex(self):
        """LLM matches that overlap regex matches are excluded."""
        text = "SSN: 123-45-6789 is on file."

        llm_entities = [
            {"type": "SSN", "value": "123-45-6789", "start": 5, "end": 16},
        ]

        with patch(
            "cam_rag.verification.sensitive.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = json.dumps(
                self._mock_ollama_response(llm_entities)
            ).encode()

            matches = scan_text(
                text, check_pii=True, check_phi=True, use_llm=True,
            )

        # Regex catches SSN. LLM's duplicate should be excluded.
        ssn_matches = [m for m in matches if "ssn" in m.pattern_name]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].pattern_name == "ssn"  # regex, not llm_ssn

    def test_llm_failure_degrades_gracefully(self):
        """If ollama is down, scan_text still returns regex results."""
        text = "SSN: 123-45-6789, Dr. Sarah Johnson."

        with patch(
            "cam_rag.verification.sensitive.urllib.request.urlopen",
            side_effect=OSError("Connection refused"),
        ):
            matches = scan_text(
                text, check_pii=True, check_phi=True, use_llm=True,
            )

        # Regex should still find SSN
        assert any(m.pattern_name == "ssn" for m in matches)
        # LLM failure should not crash
        assert not any(m.pattern_name.startswith("llm_") for m in matches)

    def test_llm_respects_category_flags(self):
        """LLM PHI matches excluded when only check_pii=True."""
        text = "Dr. Sarah Johnson, age 45, at 123 Oak St."

        llm_entities = [
            {"type": "NAME", "value": "Sarah Johnson", "start": 4, "end": 17},
            {"type": "AGE", "value": "45", "start": 23, "end": 25},
            {"type": "ADDRESS", "value": "123 Oak St", "start": 30, "end": 40},
        ]

        with patch(
            "cam_rag.verification.sensitive.urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = json.dumps(
                self._mock_ollama_response(llm_entities)
            ).encode()

            matches = scan_text(
                text, check_pii=True, check_phi=False, use_llm=True,
            )

        # NAME and AGE are PHI, should be excluded. ADDRESS is PII, included.
        assert any(m.pattern_name == "llm_address" for m in matches)
        assert not any(m.pattern_name == "llm_name" for m in matches)
        assert not any(m.pattern_name == "llm_age" for m in matches)


# ---------------------------------------------------------------------------
# Live integration test (requires ollama + model running)
# ---------------------------------------------------------------------------


@pytest.mark.llm_live
def test_live_llm_scan_healthcare_text():
    """End-to-end test with real ollama model."""
    text = (
        "Patient John Smith, age 67, was admitted on 03/15/2024. "
        "MRN: 1234567. Contact: 555-123-4567, john.smith@email.com. "
        "SSN: 123-45-6789. Dr. Sarah Johnson prescribed metoprolol. "
        "Address: 123 Oak St, Springfield IL 62704."
    )
    matches = scan_text(
        text, check_pii=True, check_phi=True, use_llm=True,
    )

    # Regex should find: SSN, email, phone, MRN
    regex_matches = [m for m in matches if not m.pattern_name.startswith("llm_")]
    assert len(regex_matches) >= 4

    # LLM should add: names (John Smith, Sarah Johnson), age, date, address
    llm_matches = [m for m in matches if m.pattern_name.startswith("llm_")]
    llm_types = {m.pattern_name for m in llm_matches}
    assert "llm_name" in llm_types, f"LLM missed NAME, got: {llm_types}"
    assert "llm_address" in llm_types or "llm_age" in llm_types, (
        f"LLM missed ADDRESS/AGE, got: {llm_types}"
    )

    # Combined redaction should cover everything
    redacted = redact_text(text, matches)
    assert "John Smith" not in redacted
    assert "123-45-6789" not in redacted
    assert "john.smith@email.com" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.llm_live
def test_live_llm_pipeline_redaction(tmp_path: Path):
    """Full pipeline with LLM redaction enabled via RAGPolicy."""
    (tmp_path / "clinical.md").write_text(
        "# Clinical Note\n\n"
        "John Smith, age 67, presented with chest pain.\n"
        "His wife Maria called from 123 Oak St, Springfield.\n"
        "Dr. Sarah Johnson evaluated the patient.\n",
        encoding="utf-8",
    )

    policy = RAGPolicy(
        enforce_pii=True,
        enforce_phi=True,
        use_llm_redaction=True,
    )
    spec = RAGAppSpec(
        name="llm-redaction-test",
        supported_extensions=(".md",),
        policy=policy,
    )
    answer = query_document_folder(tmp_path, "clinical note for patient", spec)

    # Names should be redacted (LLM catches these, regex does not)
    assert "John Smith" not in answer.answer
    assert answer.trace.confidence_details["redacted_count"] > 0
    assert "sensitive_redaction" in answer.trace.stages
