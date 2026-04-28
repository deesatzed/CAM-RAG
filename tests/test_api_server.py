"""Tests for the FastAPI HTTP/REST API server.

All tests use real query functions and real clinical text -- no mocking,
no placeholders, no cached responses.
"""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from cam_rag.api.server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Clinical text fixtures (real content, not placeholders)
# ---------------------------------------------------------------------------

_TRIAGE_DOC = {
    "id": "doc-triage-001",
    "text": (
        "Emergency department triage requires immediate assessment of vital "
        "signs including heart rate, blood pressure, respiratory rate, "
        "temperature, and oxygen saturation. The Emergency Severity Index "
        "classifies patients into five acuity levels. ESI Level 1 indicates "
        "an immediately life-threatening condition requiring resuscitation."
    ),
    "source": "ed_triage_protocol.md",
    "title": "ED Triage Protocol",
}

_SEPSIS_DOC = {
    "id": "doc-sepsis-001",
    "text": (
        "Sepsis screening criteria include systemic inflammatory response "
        "syndrome markers: temperature greater than 38C or less than 36C, "
        "heart rate greater than 90 beats per minute, respiratory rate "
        "greater than 20 breaths per minute, and white blood cell count "
        "greater than 12000 or less than 4000 per microliter. The qSOFA "
        "score is calculated using altered mental status, systolic blood "
        "pressure of 100 mmHg or less, and respiratory rate of 22 or greater."
    ),
    "source": "sepsis_screening.md",
    "title": "Sepsis Screening Protocol",
}

_BILLING_DOC = {
    "id": "doc-billing-001",
    "text": (
        "Monthly billing reconciliation ensures that invoices match "
        "charge capture. Revenue cycle management teams review denials "
        "and submit appeals within 30 days of the initial rejection."
    ),
    "source": "billing_guide.md",
    "title": "Billing Guide",
}


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/v1/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# GET /v1/config
# ---------------------------------------------------------------------------


class TestConfigEndpoint:
    def test_config_returns_feature_flags(self):
        response = client.get("/v1/config")

        assert response.status_code == 200
        body = response.json()
        assert "feature_flags" in body
        assert "policy_defaults" in body

    def test_config_feature_flag_values(self):
        response = client.get("/v1/config")
        flags = response.json()["feature_flags"]

        assert flags["retrieval_top_k"] == 10
        assert flags["dense_weight"] == 0.6
        assert flags["sparse_weight"] == 0.4
        assert flags["query_expansion_enabled"] is False
        assert flags["use_pipeline"] is False
        assert flags["use_rl"] is False
        assert flags["use_governance"] is False

    def test_config_policy_defaults(self):
        response = client.get("/v1/config")
        policy = response.json()["policy_defaults"]

        assert policy["enforce_phi"] is False
        assert policy["enforce_pii"] is False
        assert policy["require_citations"] is True
        assert policy["min_confidence"] == 0.4


# ---------------------------------------------------------------------------
# POST /v1/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_query_with_inline_documents(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "What vital signs are assessed during ED triage?",
                "documents": [_TRIAGE_DOC, _BILLING_DOC],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["confidence"] > 0
        assert len(body["evidence"]) > 0
        assert len(body["citations"]) > 0
        # The triage document should be the top result
        assert body["citations"][0]["source"] == "ed_triage_protocol.md"

    def test_query_returns_trace_information(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "sepsis screening criteria qSOFA",
                "documents": [_SEPSIS_DOC, _TRIAGE_DOC],
            },
        )

        assert response.status_code == 200
        trace = response.json()["trace"]
        assert "chunk_documents" in trace["stages"]
        assert "hybrid_rank" in trace["stages"]
        assert trace["retrieval_stats"]["documents"] == 2

    def test_query_grounding_check(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "heart rate blood pressure respiratory rate",
                "documents": [_TRIAGE_DOC, _SEPSIS_DOC],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["grounded"] is True
        assert body["confidence"] > 0

    def test_query_evidence_signals(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "ESI acuity level resuscitation",
                "documents": [_TRIAGE_DOC],
            },
        )

        assert response.status_code == 200
        evidence = response.json()["evidence"]
        assert len(evidence) > 0
        first = evidence[0]
        assert first["retriever"] == "hybrid_rrf"
        assert "matched_terms" in first["signals"]
        assert "rrf_score" in first["signals"]

    def test_query_empty_question_returns_400(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "",
                "documents": [_TRIAGE_DOC],
            },
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_query_whitespace_only_question_returns_400(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "   ",
                "documents": [_TRIAGE_DOC],
            },
        )

        assert response.status_code == 400

    def test_query_missing_documents_returns_400(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "What is triage?",
                "documents": [],
            },
        )

        assert response.status_code == 400
        assert "documents" in response.json()["detail"].lower()

    def test_query_missing_documents_field_returns_422(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "What is triage?",
            },
        )

        assert response.status_code == 422

    def test_query_with_config_overrides(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "vital signs assessment",
                "documents": [_TRIAGE_DOC, _SEPSIS_DOC, _BILLING_DOC],
                "config": {
                    "retrieval_top_k": 5,
                    "dense_weight": 0.7,
                    "sparse_weight": 0.3,
                },
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert len(body["evidence"]) > 0

    def test_query_with_query_expansion_override(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "sepsis screening markers temperature",
                "documents": [_SEPSIS_DOC, _TRIAGE_DOC],
                "config": {
                    "query_expansion_enabled": True,
                    "expansion_terms": 3,
                },
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]

    def test_query_no_evidence_for_unrelated_query(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "quantum entanglement particle physics",
                "documents": [_BILLING_DOC],
            },
        )

        assert response.status_code == 200
        body = response.json()
        # With only unrelated content, confidence should be low
        assert body["grounded"] is False
        assert body["confidence"] == 0.0

    def test_query_multiple_documents_correct_attribution(self):
        response = client.post(
            "/v1/query",
            json={
                "question": "qSOFA altered mental status systolic blood pressure",
                "documents": [_SEPSIS_DOC, _TRIAGE_DOC, _BILLING_DOC],
            },
        )

        assert response.status_code == 200
        body = response.json()
        # Sepsis document should rank highest for qSOFA query
        assert body["citations"][0]["source"] == "sepsis_screening.md"


# ---------------------------------------------------------------------------
# POST /v1/query-folder
# ---------------------------------------------------------------------------


class TestQueryFolderEndpoint:
    def test_query_folder_with_real_files(self, tmp_path: Path):
        (tmp_path / "protocol.md").write_text(
            "# Antibiotic Stewardship\n\n"
            "Empiric antibiotics should be administered within one hour of "
            "sepsis recognition. Blood cultures must be drawn before the "
            "first antibiotic dose. De-escalation to narrow-spectrum agents "
            "is required within 48 hours based on culture sensitivities.",
            encoding="utf-8",
        )
        (tmp_path / "unrelated.md").write_text(
            "# Cafeteria Menu\n\nSalad bar opens at 11:00 AM daily.",
            encoding="utf-8",
        )

        response = client.post(
            "/v1/query-folder",
            json={
                "question": "antibiotic de-escalation blood cultures",
                "docs_dir": str(tmp_path),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["citations"][0]["source"] == "protocol.md"
        assert body["grounded"] is True

    def test_query_folder_with_limit(self, tmp_path: Path):
        (tmp_path / "note.md").write_text(
            "# Clinical Note\n\nPatient presents with acute chest pain. "
            "ECG shows ST elevation in leads II, III, aVF consistent with "
            "inferior STEMI. Cardiology consulted for emergent catheterization.",
            encoding="utf-8",
        )

        response = client.post(
            "/v1/query-folder",
            json={
                "question": "STEMI ECG findings",
                "docs_dir": str(tmp_path),
                "limit": 3,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert len(body["evidence"]) <= 3

    def test_query_folder_empty_question_returns_400(self, tmp_path: Path):
        (tmp_path / "note.txt").write_text("Content here", encoding="utf-8")

        response = client.post(
            "/v1/query-folder",
            json={
                "question": "",
                "docs_dir": str(tmp_path),
            },
        )

        assert response.status_code == 400

    def test_query_folder_missing_docs_dir_returns_422(self):
        response = client.post(
            "/v1/query-folder",
            json={
                "question": "test query",
            },
        )

        assert response.status_code == 422

    def test_query_folder_empty_docs_dir_returns_400(self):
        response = client.post(
            "/v1/query-folder",
            json={
                "question": "test query",
                "docs_dir": "",
            },
        )

        assert response.status_code == 400

    def test_query_folder_nonexistent_dir_returns_400(self):
        response = client.post(
            "/v1/query-folder",
            json={
                "question": "test query",
                "docs_dir": "/nonexistent/path/that/does/not/exist",
            },
        )

        assert response.status_code == 400
        assert "not a valid directory" in response.json()["detail"]

    def test_query_folder_with_config_overrides(self, tmp_path: Path):
        (tmp_path / "vitals.md").write_text(
            "# Vital Signs\n\nHeart rate, blood pressure, respiratory rate, "
            "temperature, and oxygen saturation are standard vital signs.",
            encoding="utf-8",
        )

        response = client.post(
            "/v1/query-folder",
            json={
                "question": "vital signs monitoring",
                "docs_dir": str(tmp_path),
                "config": {
                    "retrieval_top_k": 5,
                    "query_expansion_enabled": True,
                },
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["trace"]["stages"]

    def test_query_folder_trace_includes_load_documents(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text(
            "# Discharge Planning\n\nDischarge instructions must include "
            "medication reconciliation, follow-up appointments, and "
            "warning signs requiring emergency return.",
            encoding="utf-8",
        )

        response = client.post(
            "/v1/query-folder",
            json={
                "question": "discharge medication reconciliation",
                "docs_dir": str(tmp_path),
            },
        )

        assert response.status_code == 200
        trace = response.json()["trace"]
        assert "load_documents" in trace["stages"]
        assert "chunk_documents" in trace["stages"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_json_returns_422(self):
        response = client.post(
            "/v1/query",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_unknown_endpoint_returns_404(self):
        response = client.get("/v1/nonexistent")

        assert response.status_code == 404

    def test_wrong_method_returns_405(self):
        response = client.get("/v1/query")

        assert response.status_code == 405

    def test_query_with_only_whitespace_documents_succeeds(self):
        """Documents with only whitespace text produce no evidence but no crash."""
        response = client.post(
            "/v1/query",
            json={
                "question": "something specific",
                "documents": [
                    {
                        "id": "ws-doc",
                        "text": "   ",
                        "source": "empty.md",
                        "title": "Whitespace",
                    }
                ],
            },
        )

        # The platform accepts this -- chunks get filtered by accepts_chunk
        assert response.status_code == 200
        body = response.json()
        assert body["confidence"] == 0.0
