"""Tests for PHI/PII detection, redaction, and pipeline enforcement."""

from pathlib import Path

from cam_rag.query import query_document_folder
from cam_rag.rag.spec import RAGAppSpec, RAGPolicy
from cam_rag.verification.sensitive import redact_text, scan_text

# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------


def test_pii_detection_ssn():
    text = "The patient SSN is 123-45-6789 on file."
    matches = scan_text(text, check_pii=True)
    ssn_matches = [m for m in matches if m.pattern_name == "ssn"]
    assert len(ssn_matches) == 1
    assert text[ssn_matches[0].start : ssn_matches[0].end] == "123-45-6789"


def test_pii_detection_credit_card():
    text = "Card number 4111 1111 1111 1111 was charged."
    matches = scan_text(text, check_pii=True)
    cc_matches = [m for m in matches if m.pattern_name == "credit_card"]
    assert len(cc_matches) == 1
    assert "4111" in text[cc_matches[0].start : cc_matches[0].end]


def test_pii_detection_email():
    text = "Contact admin@hospital.org for details."
    matches = scan_text(text, check_pii=True)
    email_matches = [m for m in matches if m.pattern_name == "email"]
    assert len(email_matches) == 1
    assert text[email_matches[0].start : email_matches[0].end] == "admin@hospital.org"


def test_pii_detection_phone():
    text = "Call the office at 555-867-5309 please."
    matches = scan_text(text, check_pii=True)
    phone_matches = [m for m in matches if m.pattern_name == "phone"]
    assert len(phone_matches) == 1
    assert text[phone_matches[0].start : phone_matches[0].end] == "555-867-5309"


# ---------------------------------------------------------------------------
# PHI detection
# ---------------------------------------------------------------------------


def test_phi_detection_mrn():
    text = "MRN: 12345678 was admitted today."
    matches = scan_text(text, check_phi=True)
    mrn_matches = [m for m in matches if m.pattern_name == "mrn"]
    assert len(mrn_matches) == 1
    assert "12345678" in text[mrn_matches[0].start : mrn_matches[0].end]


def test_phi_detection_dob():
    text = "DOB: 01/15/1990 recorded in chart."
    matches = scan_text(text, check_phi=True)
    dob_matches = [m for m in matches if m.pattern_name == "dob"]
    assert len(dob_matches) == 1
    assert "01/15/1990" in text[dob_matches[0].start : dob_matches[0].end]


def test_phi_detection_patient_name():
    text = "Patient: John Smith was seen in clinic."
    matches = scan_text(text, check_phi=True)
    name_matches = [m for m in matches if m.pattern_name == "patient_name"]
    assert len(name_matches) == 1
    assert "John Smith" in text[name_matches[0].start : name_matches[0].end]


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redaction_replaces_matches():
    text = "SSN 123-45-6789 and email user@test.com found."
    matches = scan_text(text, check_pii=True)
    redacted = redact_text(text, matches)
    assert "123-45-6789" not in redacted
    assert "user@test.com" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_no_redaction_when_flags_disabled():
    text = "SSN 123-45-6789 and MRN: 12345678 present."
    matches = scan_text(text, check_pii=False, check_phi=False)
    assert matches == []
    redacted = redact_text(text, matches)
    assert redacted == text


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_query_pipeline_redacts_when_enabled(tmp_path: Path):
    """When enforce_pii is True, sensitive data in the answer is redacted."""
    (tmp_path / "records.md").write_text(
        "# Records\n\n"
        "Patient SSN is 999-88-7777 and email is nurse@clinic.com.\n"
        "MRN: 00112233 DOB: 03/22/1985\n"
        "Patient: Jane Doe was discharged.",
        encoding="utf-8",
    )

    policy = RAGPolicy(enforce_pii=True, enforce_phi=True)
    spec = RAGAppSpec(
        name="phi-test",
        supported_extensions=(".md",),
        policy=policy,
    )
    answer = query_document_folder(tmp_path, "patient SSN records", spec)

    # The answer text must have redacted sensitive data
    assert "999-88-7777" not in answer.answer
    assert "nurse@clinic.com" not in answer.answer
    assert "[REDACTED]" in answer.answer
    assert answer.trace.confidence_details["redacted_count"] > 0
    assert "sensitive_redaction" in answer.trace.stages


def test_query_pipeline_no_redaction_by_default(tmp_path: Path):
    """Default policy (enforce_pii=False) does not redact."""
    (tmp_path / "records.md").write_text(
        "# Records\n\nPatient SSN is 999-88-7777.",
        encoding="utf-8",
    )

    spec = RAGAppSpec(name="default-test", supported_extensions=(".md",))
    answer = query_document_folder(tmp_path, "patient SSN records", spec)

    # Default policy should NOT redact
    assert answer.trace.confidence_details["redacted_count"] == 0
    assert "sensitive_redaction" not in answer.trace.stages
