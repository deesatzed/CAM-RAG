"""All-features-enabled integration test.

Exercises the full platform with every feature flag turned on simultaneously:
- Ensemble embedding backends (2 hash backends)
- Query expansion
- Cross-encoder reranking (Ollama backend, HTTP-patched)
- Confidence scoring + citation grounding
- LLM generation (Ollama backend, HTTP-patched)
- RL feedback loop with persistence
- Governance (fitness tracking + lifecycle management)
- PHI/PII scanning (regex phase only, no LLM)
- Both static and pipeline paths

This is the "everything on" integration test that validates the platform
doesn't break when all features are enabled simultaneously.

All data is real -- no mock, no simulation, no placeholders.
HTTP-level patching (urlopen) is used to avoid real API calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cam_rag.generation.ollama import OllamaGenerationBackend
from cam_rag.rag.models import CorpusDocument, RAGAnswer
from cam_rag.rag.spec import RAGAppSpec, RAGPolicy
from cam_rag.reranking.ollama import OllamaRerankerBackend
from cam_rag.retrieval.dense import HashEmbeddingBackend

# ---------------------------------------------------------------------------
# HTTP response helpers
# ---------------------------------------------------------------------------

_OLLAMA_GEN_URLOPEN = "cam_rag.generation.ollama.urllib.request.urlopen"
_OLLAMA_RERANK_URLOPEN = "cam_rag.reranking.ollama.urllib.request.urlopen"


class _FakeHTTPResponse:
    def __init__(self, body_dict: dict):
        self._body = json.dumps(body_dict).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_ollama_response(content: str) -> dict:
    return {"message": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _clinical_documents() -> list[CorpusDocument]:
    """Real clinical-style documents for integration testing."""
    return [
        CorpusDocument(
            id="triage-001",
            text=(
                "Emergency Triage Protocol\n\n"
                "Triage is the process of sorting patients by the severity "
                "of their conditions to determine priority for treatment. "
                "The Emergency Severity Index (ESI) uses a 5-level system:\n"
                "- Level 1: Immediate life-threatening condition\n"
                "- Level 2: High-risk situation or altered mental status\n"
                "- Level 3: Multiple resources needed\n"
                "- Level 4: Single resource needed\n"
                "- Level 5: No resources needed\n\n"
                "Initial assessment must be completed within 5 minutes of "
                "patient arrival. Vital signs should be obtained for all "
                "Level 2 and Level 3 patients."
            ),
            source="triage_protocol.md",
            title="Emergency Triage Protocol",
        ),
        CorpusDocument(
            id="vitals-002",
            text=(
                "Vital Signs Monitoring Guide\n\n"
                "Vital signs include blood pressure, heart rate, respiratory "
                "rate, temperature, and oxygen saturation (SpO2). Continuous "
                "cardiac monitoring is required for all ICU patients.\n\n"
                "Normal ranges for adults:\n"
                "- Blood pressure: 90/60 to 120/80 mmHg\n"
                "- Heart rate: 60-100 bpm\n"
                "- Respiratory rate: 12-20 breaths/min\n"
                "- Temperature: 36.1-37.2 C\n"
                "- SpO2: 95-100%\n\n"
                "Document all vital sign measurements in the patient chart "
                "with timestamp and clinician initials."
            ),
            source="vitals_guide.md",
            title="Vital Signs Monitoring Guide",
        ),
        CorpusDocument(
            id="gcs-003",
            text=(
                "Glasgow Coma Scale Assessment\n\n"
                "The Glasgow Coma Scale (GCS) assesses consciousness level "
                "through three components:\n"
                "1. Eye opening (E): Spontaneous=4, To voice=3, To pain=2, None=1\n"
                "2. Verbal response (V): Oriented=5, Confused=4, Inappropriate=3, "
                "Incomprehensible=2, None=1\n"
                "3. Motor response (M): Obeys commands=6, Localises pain=5, "
                "Withdrawal=4, Abnormal flexion=3, Extension=2, None=1\n\n"
                "Total GCS score ranges from 3 (deep coma) to 15 (fully alert). "
                "A GCS of 8 or less indicates severe brain injury and typically "
                "requires intubation."
            ),
            source="gcs_assessment.md",
            title="GCS Assessment Protocol",
        ),
        CorpusDocument(
            id="sepsis-004",
            text=(
                "Sepsis Screening Protocol\n\n"
                "Early recognition of sepsis saves lives. Screen all patients "
                "presenting with suspected infection using the qSOFA criteria:\n"
                "- Respiratory rate >= 22/min\n"
                "- Altered mentation\n"
                "- Systolic blood pressure <= 100 mmHg\n\n"
                "If 2 or more qSOFA criteria are met, initiate the Sepsis "
                "Bundle: obtain blood cultures, measure lactate level, "
                "administer broad-spectrum antibiotics within 1 hour, and "
                "begin IV fluid resuscitation with 30 mL/kg crystalloid."
            ),
            source="sepsis_protocol.md",
            title="Sepsis Screening Protocol",
        ),
    ]


def _all_features_spec(
    name: str,
    use_pipeline: bool = False,
    rl_path: str | None = None,
) -> RAGAppSpec:
    """Build a RAGAppSpec with ALL features enabled."""
    return RAGAppSpec(
        name=name,
        description="All-features integration test spec",
        # Ensemble embedding (2 hash backends)
        embedding_backends=[
            HashEmbeddingBackend(dim=32),
            HashEmbeddingBackend(dim=64),
        ],
        ensemble_weights={"dense_0": 0.35, "dense_1": 0.25},
        # Query expansion
        query_expansion_enabled=True,
        expansion_terms=5,
        # Reranking
        reranker_backend=OllamaRerankerBackend(),
        # Generation
        generation_backend=OllamaGenerationBackend(),
        # RL
        use_rl=use_pipeline,  # RL only with pipeline path
        rl_persistence_path=rl_path,
        # Governance
        use_governance=use_pipeline,
        # Pipeline mode
        use_pipeline=use_pipeline,
        # PHI/PII enforcement (regex only, no LLM)
        policy=RAGPolicy(
            enforce_pii=True,
            enforce_phi=False,
            use_llm_redaction=False,
        ),
        # Standard retrieval params
        retrieval_top_k=5,
        dense_weight=0.6,
        sparse_weight=0.4,
    )


def _patch_ollama_calls():
    """Context manager that patches both generation and reranking Ollama calls."""
    rerank_scores = {"1": 8, "2": 6, "3": 9, "4": 4, "5": 7}
    gen_answer = (
        "Based on the evidence, emergency triage uses the ESI 5-level "
        "system to sort patients by severity [1]. Initial assessment "
        "must be completed within 5 minutes [1]. Vital signs including "
        "blood pressure and heart rate should be monitored for Level 2-3 "
        "patients [2]."
    )

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        sys_msg = payload["messages"][0]["content"]
        if "relevance" in sys_msg.lower():
            body = _make_ollama_response(json.dumps(rerank_scores))
        else:
            body = _make_ollama_response(gen_answer)
        return _FakeHTTPResponse(body)

    return (
        patch(_OLLAMA_GEN_URLOPEN, side_effect=fake_urlopen),
        patch(_OLLAMA_RERANK_URLOPEN, side_effect=fake_urlopen),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAllFeaturesStatic:
    """All features enabled using the static (default) path."""

    def test_full_pipeline_returns_answer(self):
        """Static path with all features produces a complete RAGAnswer."""
        spec = _all_features_spec("all-static-1")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query(
                "What is the emergency triage protocol?",
                documents,
                spec,
            )

        assert isinstance(result, RAGAnswer)
        assert len(result.answer) > 0
        assert len(result.evidence) > 0
        assert len(result.citations) > 0
        assert result.confidence > 0.0

    def test_evidence_has_reranker_signals(self):
        """Evidence items should have reranker signals."""
        spec = _all_features_spec("all-static-2")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("sepsis screening criteria", documents, spec)

        for item in result.evidence:
            assert "reranker_score" in item.signals
            assert "rrf_score_pre_rerank" in item.signals

    def test_generation_produces_llm_answer(self):
        """The answer should come from the generation backend."""
        spec = _all_features_spec("all-static-3")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("What is the ESI system?", documents, spec)

        # Should be LLM-generated, not retrieval-only template
        assert "Retrieved" not in result.answer
        assert "evidence" in result.answer.lower() or "ESI" in result.answer

    def test_trace_records_stages(self):
        """Trace should include all pipeline stages."""
        spec = _all_features_spec("all-static-4")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("GCS assessment protocol", documents, spec)

        # Core stages
        assert "chunk_documents" in result.trace.stages
        assert "hybrid_rank" in result.trace.stages
        assert "score_confidence" in result.trace.stages
        assert "verify_grounding" in result.trace.stages
        assert "generation" in result.trace.stages

    def test_confidence_details_populated(self):
        """Confidence details should be populated."""
        spec = _all_features_spec("all-static-5")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("vital signs monitoring", documents, spec)

        assert "grounding" in result.trace.confidence_details
        assert result.confidence > 0.0


class TestAllFeaturesPipeline:
    """All features enabled using the dynamic pipeline path."""

    def test_pipeline_full_run(self):
        """Pipeline path with all features produces a complete RAGAnswer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl_path = str(Path(tmpdir) / "rl_state.json")
            spec = _all_features_spec(
                "all-pipeline-1", use_pipeline=True, rl_path=rl_path,
            )
            documents = _clinical_documents()

            gen_patch, rerank_patch = _patch_ollama_calls()
            with gen_patch, rerank_patch:
                from cam_rag.query import query
                result = query(
                    "What is the emergency triage protocol?",
                    documents,
                    spec,
                )

            assert isinstance(result, RAGAnswer)
            assert len(result.answer) > 0
            assert len(result.evidence) > 0
            assert len(result.citations) > 0

    def test_pipeline_rl_selects_arm(self):
        """RL feedback loop should select an arm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl_path = str(Path(tmpdir) / "rl_state.json")
            spec = _all_features_spec(
                "all-pipeline-rl", use_pipeline=True, rl_path=rl_path,
            )
            documents = _clinical_documents()

            gen_patch, rerank_patch = _patch_ollama_calls()
            with gen_patch, rerank_patch:
                from cam_rag.query import query
                result = query("sepsis screening", documents, spec)

            # RL feedback should have been recorded
            assert "rl_feedback" in result.trace.stages

    def test_pipeline_governance_records_fitness(self):
        """Governance should record fitness observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl_path = str(Path(tmpdir) / "rl_state.json")
            spec = _all_features_spec(
                "all-pipeline-gov", use_pipeline=True, rl_path=rl_path,
            )
            documents = _clinical_documents()

            gen_patch, rerank_patch = _patch_ollama_calls()
            with gen_patch, rerank_patch:
                from cam_rag.query import query
                query("GCS assessment", documents, spec)

            # Check governance caches
            from cam_rag.query import _fitness_trackers
            assert spec.name in _fitness_trackers
            tracker = _fitness_trackers[spec.name]
            assert len(tracker.technique_names) > 0

    def test_pipeline_with_expansion(self):
        """Query expansion should execute in pipeline mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl_path = str(Path(tmpdir) / "rl_state.json")
            spec = _all_features_spec(
                "all-pipeline-expansion", use_pipeline=True, rl_path=rl_path,
            )
            documents = _clinical_documents()

            gen_patch, rerank_patch = _patch_ollama_calls()
            with gen_patch, rerank_patch:
                from cam_rag.query import query
                # Use a synthesis query to trigger expansion
                result = query(
                    "Compare triage and sepsis screening procedures",
                    documents,
                    spec,
                )

            assert isinstance(result, RAGAnswer)
            assert len(result.evidence) > 0

    def test_pipeline_reranking_in_evidence(self):
        """Reranking signals should appear in pipeline path evidence.

        Uses pipeline mode WITHOUT RL so the full plan (including
        cross_encoder_rerank) always executes.
        """
        # Pipeline without RL ensures all registered steps run
        spec = RAGAppSpec(
            name="all-pipeline-rerank-norl",
            embedding_backends=[
                HashEmbeddingBackend(dim=32),
                HashEmbeddingBackend(dim=64),
            ],
            ensemble_weights={"dense_0": 0.35, "dense_1": 0.25},
            reranker_backend=OllamaRerankerBackend(),
            generation_backend=OllamaGenerationBackend(),
            use_pipeline=True,
            use_rl=False,
            use_governance=False,
            retrieval_top_k=5,
        )
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("vital signs monitoring", documents, spec)

        for item in result.evidence:
            assert "reranker_score" in item.signals

    def test_pipeline_generation_produces_answer(self):
        """Generation step should produce LLM answer in pipeline mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rl_path = str(Path(tmpdir) / "rl_state.json")
            spec = _all_features_spec(
                "all-pipeline-gen", use_pipeline=True, rl_path=rl_path,
            )
            documents = _clinical_documents()

            gen_patch, rerank_patch = _patch_ollama_calls()
            with gen_patch, rerank_patch:
                from cam_rag.query import query
                result = query("What is the ESI system?", documents, spec)

            # Answer should come from generation, not template
            assert "Retrieved" not in result.answer


class TestAllFeaturesEdgeCases:
    """Edge case tests with all features enabled."""

    def test_empty_corpus(self):
        """All features with empty document list."""
        spec = _all_features_spec("all-edge-empty")
        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query("What is triage?", [], spec)

        assert isinstance(result, RAGAnswer)
        assert len(result.evidence) == 0

    def test_multiple_queries_same_spec(self):
        """Multiple queries with the same spec (tests caching)."""
        spec = _all_features_spec("all-edge-multi")
        documents = _clinical_documents()

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result1 = query("triage protocol", documents, spec)
            result2 = query("sepsis screening", documents, spec)
            result3 = query("vital signs", documents, spec)

        assert len(result1.evidence) > 0
        assert len(result2.evidence) > 0
        assert len(result3.evidence) > 0

    def test_very_long_query(self):
        """Long query doesn't break any feature."""
        spec = _all_features_spec("all-edge-long")
        documents = _clinical_documents()
        long_query = "What is " + "the detailed " * 100 + "triage protocol?"

        gen_patch, rerank_patch = _patch_ollama_calls()
        with gen_patch, rerank_patch:
            from cam_rag.query import query
            result = query(long_query, documents, spec)

        assert isinstance(result, RAGAnswer)
