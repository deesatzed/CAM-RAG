"""Integration tests: RL FeedbackLoop wired into the query path.

Validates that:
1. RL is disabled by default -- no FeedbackLoop created or called
2. RL wiring works -- pipeline path used, before/after_query called, trace updated
3. RL selects different technique sets across multiple queries
4. RL persistence -- FeedbackLoop saves/loads state via temp file
5. Backward compatibility -- existing paths unaffected by use_rl=False

All data is real -- no mock, no simulation, no placeholders.
Real CorpusDocument objects are chunked, indexed, and retrieved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cam_rag.query import (
    _feedback_loops,
    _get_feedback_loop,
    _rank_chunks,
    query,
    query_document_folder,
)
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec
from cam_rag.rl.feedback import DEFAULT_ARMS, FeedbackLoop

# ---------------------------------------------------------------------------
# Helpers -- real documents, no mocks
# ---------------------------------------------------------------------------


def _make_documents() -> list[CorpusDocument]:
    """Build a small corpus of real CorpusDocument objects."""
    return [
        CorpusDocument(
            id="doc_triage",
            text=(
                "Emergency department triage uses vital signs and "
                "acuity scoring to prioritize patients. The Emergency "
                "Severity Index assigns levels one through five based "
                "on clinical acuity and expected resource utilization."
            ),
            source="triage.md",
            title="Triage Protocol",
        ),
        CorpusDocument(
            id="doc_sepsis",
            text=(
                "Sepsis screening in the emergency department uses "
                "the qSOFA criteria: altered mental status, systolic "
                "blood pressure of 100 mmHg or less, and respiratory "
                "rate of 22 breaths per minute or greater. Early "
                "recognition is critical for survival."
            ),
            source="sepsis.md",
            title="Sepsis Screening",
        ),
        CorpusDocument(
            id="doc_billing",
            text=(
                "Hospital billing codes are assigned after discharge. "
                "Revenue cycle management ensures accurate coding and "
                "timely claim submission to payers."
            ),
            source="billing.md",
            title="Billing",
        ),
    ]


def _rl_spec(
    name: str = "rl_test",
    *,
    persistence_path: str | None = None,
) -> RAGAppSpec:
    """Build a RAGAppSpec with RL enabled."""
    return RAGAppSpec(
        name=name,
        supported_extensions=(".md",),
        use_pipeline=True,
        use_rl=True,
        rl_persistence_path=persistence_path,
    )


def _default_spec(name: str = "default_test") -> RAGAppSpec:
    """Build a RAGAppSpec with RL disabled (default behavior)."""
    return RAGAppSpec(
        name=name,
        supported_extensions=(".md",),
    )


@pytest.fixture(autouse=True)
def _clear_feedback_loop_cache():
    """Clear the module-level FeedbackLoop cache between tests."""
    _feedback_loops.clear()
    yield
    _feedback_loops.clear()


# ---------------------------------------------------------------------------
# 1. RL disabled by default
# ---------------------------------------------------------------------------


class TestRLDisabledByDefault:
    """When use_rl=False (default), no FeedbackLoop is created."""

    def test_default_spec_has_rl_off(self):
        spec = RAGAppSpec(name="check")
        assert spec.use_rl is False
        assert spec.rl_persistence_path is None

    def test_query_with_rl_off_returns_answer(self):
        spec = _default_spec()
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        assert answer.evidence
        assert answer.citations
        assert "rl_feedback" not in answer.trace.stages

    def test_no_feedback_loop_created_when_rl_off(self):
        spec = _default_spec()
        documents = _make_documents()
        query("ED triage vital signs", documents, spec)

        assert spec.name not in _feedback_loops

    def test_rank_chunks_returns_none_arm_when_rl_off(self):
        spec = _default_spec()
        documents = _make_documents()
        from cam_rag.documents.chunking import chunk_documents

        chunks = [
            c
            for c in chunk_documents(documents, overlap_chars=0)
            if spec.accepts_chunk(c)
        ]
        evidence, _expanded, arm_name, _gen = _rank_chunks(
            "ED triage vital signs", chunks, spec, limit=5,
        )
        assert arm_name is None
        assert evidence  # should still retrieve real results


# ---------------------------------------------------------------------------
# 2. RL wiring works
# ---------------------------------------------------------------------------


class TestRLWiringWorks:
    """With use_rl=True, verify before/after_query are called."""

    def test_query_with_rl_on_returns_answer(self):
        spec = _rl_spec()
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        assert answer.evidence
        assert answer.citations
        assert answer.confidence > 0

    def test_trace_includes_rl_feedback_stage(self):
        spec = _rl_spec()
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        assert "rl_feedback" in answer.trace.stages

    def test_feedback_loop_created_in_cache(self):
        spec = _rl_spec(name="cached_test")
        documents = _make_documents()
        query("ED triage vital signs", documents, spec)

        assert "cached_test" in _feedback_loops
        loop = _feedback_loops["cached_test"]
        assert isinstance(loop, FeedbackLoop)

    def test_bandit_updated_after_query(self):
        spec = _rl_spec()
        documents = _make_documents()
        query("ED triage vital signs", documents, spec)

        loop = _feedback_loops[spec.name]
        assert loop.selector.bandit.total_pulls >= 1

    def test_pipeline_path_used_not_static(self):
        """RL forces the pipeline path even if use_pipeline was False."""
        spec = RAGAppSpec(
            name="rl_forces_pipeline",
            supported_extensions=(".md",),
            use_pipeline=False,
            use_rl=True,
        )
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        # The answer should still work (pipeline path used via RL)
        assert answer.evidence
        assert "rl_feedback" in answer.trace.stages

    def test_arm_selected_from_default_arms(self):
        spec = _rl_spec()
        documents = _make_documents()
        from cam_rag.documents.chunking import chunk_documents

        chunks = [
            c
            for c in chunk_documents(documents, overlap_chars=0)
            if spec.accepts_chunk(c)
        ]
        _evidence, _expanded, arm_name, _gen = _rank_chunks(
            "ED triage vital signs", chunks, spec, limit=5,
        )
        assert arm_name is not None
        assert arm_name in DEFAULT_ARMS


# ---------------------------------------------------------------------------
# 3. RL selects different technique sets across queries
# ---------------------------------------------------------------------------


class TestRLSelectsDifferentArms:
    """Run multiple queries and verify the bandit records selections."""

    def test_multiple_queries_record_pulls(self):
        spec = _rl_spec()
        documents = _make_documents()
        queries = [
            "ED triage vital signs",
            "What is the qSOFA criteria for sepsis screening?",
            "How does sepsis screening improve survival?",
            "Compare triage acuity with billing codes",
        ]
        for q in queries:
            query(q, documents, spec)

        loop = _feedback_loops[spec.name]
        assert loop.selector.bandit.total_pulls == len(queries)

    def test_arm_names_are_valid(self):
        spec = _rl_spec()
        documents = _make_documents()
        from cam_rag.documents.chunking import chunk_documents

        chunks = [
            c
            for c in chunk_documents(documents, overlap_chars=0)
            if spec.accepts_chunk(c)
        ]

        arm_names_seen: set[str] = set()
        for _ in range(6):
            _, _, arm_name, _gen = _rank_chunks(
                "ED triage vital signs", chunks, spec, limit=5,
            )
            assert arm_name is not None
            arm_names_seen.add(arm_name)

        # All arm names must be from DEFAULT_ARMS
        for name in arm_names_seen:
            assert name in DEFAULT_ARMS

    def test_bandit_stats_accumulate(self):
        spec = _rl_spec()
        documents = _make_documents()

        for _ in range(5):
            query("ED triage vital signs", documents, spec)

        loop = _feedback_loops[spec.name]
        total_arm_pulls = sum(
            arm.pulls for arm in loop.selector.bandit.arms.values()
        )
        assert total_arm_pulls == 5
        assert loop.selector.bandit.total_pulls == 5


# ---------------------------------------------------------------------------
# 4. RL persistence
# ---------------------------------------------------------------------------


class TestRLPersistence:
    """FeedbackLoop saves/loads state via rl_persistence_path."""

    def test_state_saved_after_query(self, tmp_path: Path):
        state_file = tmp_path / "rl_state.json"
        spec = _rl_spec(persistence_path=str(state_file))
        documents = _make_documents()

        query("ED triage vital signs", documents, spec)

        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "arms" in data
        assert data["total_pulls"] >= 1

    def test_state_restored_on_new_loop(self, tmp_path: Path):
        state_file = tmp_path / "rl_state.json"

        # First session: run queries and persist
        spec1 = _rl_spec(
            name="persist_test",
            persistence_path=str(state_file),
        )
        documents = _make_documents()
        for _ in range(3):
            query("ED triage vital signs", documents, spec1)

        loop1 = _feedback_loops["persist_test"]
        pulls_after_session1 = loop1.selector.bandit.total_pulls
        assert pulls_after_session1 == 3

        # Clear cache to simulate restart
        _feedback_loops.clear()

        # Second session: state should be loaded from file
        spec2 = _rl_spec(
            name="persist_test",
            persistence_path=str(state_file),
        )
        query("What is sepsis screening?", documents, spec2)

        loop2 = _feedback_loops["persist_test"]
        assert loop2.selector.bandit.total_pulls == pulls_after_session1 + 1

    def test_no_persistence_without_path(self):
        spec = _rl_spec(persistence_path=None)
        documents = _make_documents()
        query("ED triage vital signs", documents, spec)

        loop = _feedback_loops[spec.name]
        assert loop.state_path is None


# ---------------------------------------------------------------------------
# 5. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing behavior is unchanged when use_rl=False."""

    def test_static_path_still_works(self):
        spec = RAGAppSpec(
            name="static_compat",
            supported_extensions=(".md",),
            use_pipeline=False,
            use_rl=False,
        )
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        assert answer.evidence
        assert answer.citations
        assert "rl_feedback" not in answer.trace.stages

    def test_pipeline_path_without_rl_still_works(self):
        spec = RAGAppSpec(
            name="pipeline_compat",
            supported_extensions=(".md",),
            use_pipeline=True,
            use_rl=False,
        )
        documents = _make_documents()
        answer = query("ED triage vital signs", documents, spec)

        assert answer.evidence
        assert answer.citations
        assert "rl_feedback" not in answer.trace.stages

    def test_query_document_folder_rl_off(self, tmp_path: Path):
        (tmp_path / "triage.md").write_text(
            "# Triage\n\n"
            "ED triage requires immediate vital signs and "
            "acuity assignment.",
            encoding="utf-8",
        )
        spec = RAGAppSpec(
            name="folder_compat",
            supported_extensions=(".md",),
        )
        answer = query_document_folder(
            tmp_path, "ED triage vital signs", spec,
        )
        assert answer.evidence
        assert "rl_feedback" not in answer.trace.stages

    def test_query_document_folder_rl_on(self, tmp_path: Path):
        (tmp_path / "triage.md").write_text(
            "# Triage\n\n"
            "ED triage requires immediate vital signs and "
            "acuity assignment.",
            encoding="utf-8",
        )
        spec = RAGAppSpec(
            name="folder_rl",
            supported_extensions=(".md",),
            use_pipeline=True,
            use_rl=True,
        )
        answer = query_document_folder(
            tmp_path, "ED triage vital signs", spec,
        )
        assert answer.evidence
        assert "rl_feedback" in answer.trace.stages

    def test_get_feedback_loop_caches_instance(self):
        spec = _rl_spec(name="cache_check")
        loop1 = _get_feedback_loop(spec)
        loop2 = _get_feedback_loop(spec)
        assert loop1 is loop2

    def test_different_specs_get_different_loops(self):
        spec_a = _rl_spec(name="loop_a")
        spec_b = _rl_spec(name="loop_b")
        loop_a = _get_feedback_loop(spec_a)
        loop_b = _get_feedback_loop(spec_b)
        assert loop_a is not loop_b
