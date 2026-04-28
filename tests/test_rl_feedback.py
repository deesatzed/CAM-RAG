"""Tests for RL feedback loop (RewardComputer, BanditSelector, FeedbackLoop).

Validates that:
1. RewardComputer produces correct rewards from real RAGAnswer objects
2. BanditSelector cold-starts, selects valid technique sets, records rewards
3. FeedbackLoop ties selection + reward + persistence together
4. Integration: full loop with a real RAGAnswer through the pipeline

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cam_rag.pipeline.registry import (
    TechniqueDescriptor,
    TechniqueRegistry,
    compose_pipeline,
)
from cam_rag.rag.models import (
    Chunk,
    Citation,
    Evidence,
    RAGAnswer,
    RAGTrace,
)
from cam_rag.rl.bandit import UCB1Bandit
from cam_rag.rl.feedback import (
    DEFAULT_ARMS,
    BanditSelector,
    FeedbackLoop,
    RewardComputer,
)

# ---------------------------------------------------------------------------
# Fixtures -- real objects, no mocks
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "chunk_0",
    doc_id: str = "doc_0",
    text: str = "Emergency department triage uses vital signs.",
    source: str = "triage.md",
    title: str = "Triage",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        text=text,
        source=source,
        title=title,
    )


def _make_evidence(
    n: int = 3,
    score: float = 0.025,
) -> list[Evidence]:
    """Build a list of real Evidence objects with realistic signals."""
    texts = [
        "Emergency department triage uses vital signs and acuity scoring.",
        "Chest pain evaluation follows the HEART score algorithm.",
        "Sepsis screening uses the qSOFA criteria.",
    ]
    evidence: list[Evidence] = []
    for i in range(n):
        chunk = _make_chunk(
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            text=texts[i % len(texts)],
            source=f"document_{i}.md",
            title=f"Section {i}",
        )
        evidence.append(
            Evidence(
                chunk=chunk,
                score=score * (n - i),
                retriever="hybrid_rrf",
                rank=i,
                signals={
                    "matched_terms": ["triage", "vital"],
                    "source_ranks": {"dense": i + 1, "sparse": i + 2},
                    "source_scores": {
                        "dense": 0.9 - i * 0.1,
                        "sparse": 0.8 - i * 0.1,
                    },
                    "rrf_score": score * (n - i),
                },
            )
        )
    return evidence


def _make_citations(evidence: list[Evidence]) -> list[Citation]:
    return [
        Citation(
            source=e.chunk.source,
            document_id=e.chunk.document_id,
            title=e.chunk.title,
            excerpt=e.chunk.text[:120],
            score=e.score,
        )
        for e in evidence
    ]


def _make_trace(
    overall: float = 0.85,
    grounded: bool = True,
) -> RAGTrace:
    """Build a real RAGTrace with confidence_details populated."""
    grounding_data = {
        "grounded": grounded,
        "checked_citations": 3,
        "matched_citations": 3 if grounded else 0,
        "missing_sources": [] if grounded else ["missing.md"],
    }
    return RAGTrace(
        query_type="specification",
        stages=[
            "chunk_documents", "hybrid_rank", "score_confidence",
        ],
        confidence_details={
            "overall": overall,
            "top_score": 0.075,
            "score_margin": 0.025,
            "evidence_count": 3,
            "source_agreement": 1.0,
            "query_coverage": 0.5,
            "grounding": grounding_data,
        },
    )


def _make_answer(
    confidence: float = 0.85,
    grounded: bool = True,
    evidence_count: int = 3,
) -> RAGAnswer:
    """Build a fully-populated real RAGAnswer."""
    evidence = _make_evidence(n=evidence_count)
    citations = _make_citations(evidence)
    trace = _make_trace(overall=confidence, grounded=grounded)
    return RAGAnswer(
        answer="Triage scoring uses vital signs and acuity protocols.",
        evidence=evidence,
        citations=citations,
        confidence=confidence,
        grounded=grounded,
        trace=trace,
    )


def _build_default_registry() -> TechniqueRegistry:
    """Build the default technique registry (mirrors query.py)."""
    registry = TechniqueRegistry()
    techniques = [
        TechniqueDescriptor(
            name="sparse_bm25", category="retrieval",
        ),
        TechniqueDescriptor(
            name="dense_vector", category="retrieval",
        ),
        TechniqueDescriptor(
            name="rrf_fusion",
            category="retrieval",
            requires=("sparse_bm25", "dense_vector"),
        ),
        TechniqueDescriptor(
            name="query_expansion",
            category="expansion",
            requires=("rrf_fusion",),
            query_types=("summary", "synthesis"),
        ),
        TechniqueDescriptor(
            name="confidence_scoring",
            category="verification",
        ),
        TechniqueDescriptor(
            name="citation_grounding",
            category="verification",
            requires=("confidence_scoring",),
        ),
    ]
    for t in techniques:
        registry.register(t)
    return registry


# -------------------------------------------------------------------
# 1. RewardComputer
# -------------------------------------------------------------------


class TestRewardComputerHighConfidence:
    """Reward from a high-confidence, grounded answer with evidence."""

    def test_high_confidence_grounded_full_evidence(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.9, grounded=True, evidence_count=3,
        )
        reward = computer.compute(answer)

        assert 0.0 <= reward <= 1.0
        # 0.9*0.5 + 1.0*0.3 + 1.0*0.2
        expected = 0.9 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2
        assert abs(reward - expected) < 1e-9

    def test_perfect_answer_yields_max_reward(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=1.0, grounded=True, evidence_count=5,
        )
        reward = computer.compute(answer)
        assert reward == 1.0


class TestRewardComputerLowConfidence:
    """Reward from a low-confidence answer."""

    def test_low_confidence_ungrounded(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.1, grounded=False, evidence_count=1,
        )
        reward = computer.compute(answer)

        assert 0.0 <= reward <= 1.0
        expected = 0.1 * 0.5 + 0.0 * 0.3 + (1 / 3) * 0.2
        assert abs(reward - expected) < 1e-9

    def test_low_confidence_still_grounded(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.2, grounded=True, evidence_count=2,
        )
        reward = computer.compute(answer)
        assert reward > 0.0
        assert reward > 0.2 * 0.5


class TestRewardComputerEmptyEvidence:
    """Reward when no evidence is retrieved."""

    def test_empty_evidence_zero_evidence_signal(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.5, grounded=False, evidence_count=0,
        )
        reward = computer.compute(answer)

        assert 0.0 <= reward <= 1.0
        expected = 0.5 * 0.5
        assert abs(reward - expected) < 1e-9

    def test_no_evidence_no_grounding_low_confidence(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.0, grounded=False, evidence_count=0,
        )
        reward = computer.compute(answer)
        assert reward == 0.0


class TestRewardComputerConfidenceFromTrace:
    """RewardComputer prefers confidence from trace details."""

    def test_uses_trace_overall_over_top_level(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.3, grounded=True, evidence_count=3,
        )
        answer.confidence = 0.99
        answer.trace.confidence_details["overall"] = 0.3
        reward = computer.compute(answer)

        expected = 0.3 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2
        assert abs(reward - expected) < 1e-9

    def test_falls_back_to_top_level_when_no_details(self):
        computer = RewardComputer()
        answer = _make_answer(
            confidence=0.7, grounded=True, evidence_count=3,
        )
        answer.trace.confidence_details = {}
        reward = computer.compute(answer)

        expected = 0.7 * 0.5 + 1.0 * 0.3 + 1.0 * 0.2
        assert abs(reward - expected) < 1e-9


class TestRewardComputerCustomWeights:
    """RewardComputer respects custom weights."""

    def test_confidence_only_weight(self):
        computer = RewardComputer(
            confidence_weight=1.0,
            grounding_weight=0.0,
            evidence_weight=0.0,
        )
        answer = _make_answer(
            confidence=0.6, grounded=True, evidence_count=3,
        )
        reward = computer.compute(answer)
        assert abs(reward - 0.6) < 1e-9

    def test_evidence_only_weight(self):
        computer = RewardComputer(
            confidence_weight=0.0,
            grounding_weight=0.0,
            evidence_weight=1.0,
            evidence_target=5,
        )
        answer = _make_answer(
            confidence=0.9, grounded=True, evidence_count=3,
        )
        reward = computer.compute(answer)
        assert abs(reward - 3 / 5) < 1e-9


# -------------------------------------------------------------------
# 2. BanditSelector
# -------------------------------------------------------------------


class TestBanditSelectorColdStart:
    """BanditSelector registers default arms on first use."""

    def test_cold_start_registers_default_arms(self):
        selector = BanditSelector()
        arm_name, techniques = selector.select_techniques(
            "specification", corpus_size=10,
        )

        assert arm_name in DEFAULT_ARMS
        assert len(selector.bandit.arms) == len(DEFAULT_ARMS)
        assert techniques == DEFAULT_ARMS[arm_name]

    def test_cold_start_only_happens_once(self):
        selector = BanditSelector()
        selector.select_techniques("specification", corpus_size=10)
        initial_count = len(selector.bandit.arms)

        selector.select_techniques("summary", corpus_size=10)
        assert len(selector.bandit.arms) == initial_count

    def test_cold_start_skipped_if_arms_present(self):
        bandit = UCB1Bandit()
        bandit.register_arm("custom_arm")
        selector = BanditSelector(
            bandit=bandit,
            arm_techniques={"custom_arm": {"sparse_bm25"}},
        )
        arm_name, _techniques = selector.select_techniques(
            "specification", corpus_size=10,
        )
        assert arm_name == "custom_arm"
        assert len(selector.bandit.arms) == 1


class TestBanditSelectorSelection:
    """BanditSelector returns valid technique sets."""

    def test_selection_returns_valid_technique_set(self):
        selector = BanditSelector()
        _arm_name, techniques = selector.select_techniques(
            "summary", corpus_size=10,
        )
        assert isinstance(techniques, set)
        assert len(techniques) > 0
        assert "sparse_bm25" in techniques

    def test_selection_with_registry_filters(self):
        selector = BanditSelector()
        registry = _build_default_registry()
        _arm_name, techniques = selector.select_techniques(
            "specification",
            corpus_size=10,
            registry=registry,
        )
        assert "query_expansion" not in techniques
        assert "sparse_bm25" in techniques

    def test_selection_with_registry_summary_expansion(self):
        selector = BanditSelector()
        registry = _build_default_registry()

        selector.select_techniques("summary", corpus_size=10)
        selector.record_reward("base_retrieval", 0.1)
        selector.record_reward("base+expansion", 0.9)
        selector.record_reward("full_pipeline", 0.1)

        for _ in range(10):
            arm_name, techniques = selector.select_techniques(
                "summary", corpus_size=10, registry=registry,
            )
            reward = 0.8 if "expansion" in arm_name else 0.2
            selector.record_reward(arm_name, reward)

        arm_name, techniques = selector.select_techniques(
            "summary", corpus_size=10, registry=registry,
        )
        if arm_name in ("base+expansion", "full_pipeline"):
            assert "query_expansion" in techniques


class TestBanditSelectorRecordReward:
    """Recording reward updates arm stats."""

    def test_record_reward_updates_stats(self):
        selector = BanditSelector()
        selector.select_techniques("specification", corpus_size=10)

        first_arm = next(iter(selector.bandit.arms.keys()))
        selector.record_reward(first_arm, 0.8)

        assert selector.bandit.arms[first_arm].pulls == 1
        assert selector.bandit.arms[first_arm].mean_reward == 0.8
        assert selector.bandit.total_pulls == 1

    def test_record_reward_invalid_arm_raises(self):
        selector = BanditSelector()
        with pytest.raises(KeyError, match="Unknown arm"):
            selector.record_reward("nonexistent", 0.5)

    def test_record_reward_out_of_range_raises(self):
        selector = BanditSelector()
        selector.select_techniques("specification", corpus_size=10)
        first_arm = next(iter(selector.bandit.arms.keys()))

        with pytest.raises(ValueError, match="Reward must be"):
            selector.record_reward(first_arm, 1.5)


class TestBanditSelectorCustomArm:
    """Register and use custom arms."""

    def test_register_custom_arm(self):
        selector = BanditSelector()
        selector.register_arm(
            "sparse_only",
            {"sparse_bm25", "confidence_scoring"},
        )
        assert "sparse_only" in selector.bandit.arm_names
        assert selector.arm_techniques["sparse_only"] == {
            "sparse_bm25",
            "confidence_scoring",
        }

    def test_custom_arm_selected_when_only_option(self):
        selector = BanditSelector()
        selector.register_arm("only_arm", {"sparse_bm25"})
        arm_name, techniques = selector.select_techniques(
            "specification", corpus_size=10,
        )
        assert arm_name == "only_arm"
        assert techniques == {"sparse_bm25"}


# -------------------------------------------------------------------
# 3. BanditSelector serialisation
# -------------------------------------------------------------------


class TestBanditSelectorSerialisation:
    """Round-trip serialisation to/from dict."""

    def test_round_trip(self):
        selector = BanditSelector()
        selector.select_techniques("specification", corpus_size=10)
        for arm_name in list(selector.bandit.arms.keys()):
            selector.record_reward(arm_name, 0.5)

        data = selector.to_dict()
        restored = BanditSelector.from_dict(data)

        assert (
            restored.bandit.total_pulls
            == selector.bandit.total_pulls
        )
        assert set(restored.bandit.arms.keys()) == set(
            selector.bandit.arms.keys(),
        )
        for name in selector.bandit.arms:
            orig = selector.bandit.arms[name]
            rest = restored.bandit.arms[name]
            assert orig.pulls == rest.pulls
            assert abs(orig.mean_reward - rest.mean_reward) < 1e-12
            assert (
                restored.arm_techniques[name]
                == selector.arm_techniques[name]
            )

    def test_to_dict_structure(self):
        selector = BanditSelector()
        selector.register_arm(
            "test_arm", {"sparse_bm25", "dense_vector"},
        )
        selector.record_reward("test_arm", 0.7)

        data = selector.to_dict()
        assert "total_pulls" in data
        assert "exploration_weight" in data
        assert "arms" in data
        assert "test_arm" in data["arms"]

        arm_data = data["arms"]["test_arm"]
        assert arm_data["pulls"] == 1
        assert abs(arm_data["mean_reward"] - 0.7) < 1e-12
        assert sorted(arm_data["techniques"]) == [
            "dense_vector",
            "sparse_bm25",
        ]


# -------------------------------------------------------------------
# 4. FeedbackLoop
# -------------------------------------------------------------------


class TestFeedbackLoopBeforeQuery:
    """FeedbackLoop.before_query returns techniques."""

    def test_returns_arm_and_techniques(self):
        loop = FeedbackLoop()
        arm_name, techniques = loop.before_query(
            "specification", corpus_size=10,
        )
        assert arm_name in DEFAULT_ARMS
        assert isinstance(techniques, set)
        assert len(techniques) > 0

    def test_with_registry(self):
        loop = FeedbackLoop()
        registry = _build_default_registry()
        _arm_name, techniques = loop.before_query(
            "specification",
            corpus_size=10,
            registry=registry,
        )
        assert isinstance(techniques, set)
        assert "query_expansion" not in techniques


class TestFeedbackLoopAfterQuery:
    """FeedbackLoop.after_query computes reward and updates state."""

    def test_after_query_returns_reward(self):
        loop = FeedbackLoop()
        arm_name, _ = loop.before_query(
            "specification", corpus_size=10,
        )
        answer = _make_answer(
            confidence=0.8, grounded=True, evidence_count=3,
        )
        reward = loop.after_query(arm_name, answer)
        assert 0.0 <= reward <= 1.0

    def test_after_query_updates_bandit(self):
        loop = FeedbackLoop()
        arm_name, _ = loop.before_query(
            "specification", corpus_size=10,
        )
        answer = _make_answer(
            confidence=0.7, grounded=True, evidence_count=2,
        )
        loop.after_query(arm_name, answer)
        assert loop.selector.bandit.arms[arm_name].pulls == 1
        assert loop.selector.bandit.total_pulls == 1

    def test_multiple_rounds_update_bandit(self):
        loop = FeedbackLoop()
        for i in range(5):
            arm_name, _ = loop.before_query(
                "summary", corpus_size=10,
            )
            conf = 0.5 + i * 0.1
            answer = _make_answer(
                confidence=conf,
                grounded=True,
                evidence_count=3,
            )
            loop.after_query(arm_name, answer)

        assert loop.selector.bandit.total_pulls == 5


class TestFeedbackLoopPersistence:
    """FeedbackLoop JSON persistence round-trip."""

    def test_save_and_load(self, tmp_path: Path):
        state_file = tmp_path / "bandit_state.json"
        loop = FeedbackLoop(state_path=state_file)

        arm_name, _ = loop.before_query(
            "specification", corpus_size=10,
        )
        answer = _make_answer(
            confidence=0.8, grounded=True, evidence_count=3,
        )
        loop.after_query(arm_name, answer)

        assert state_file.exists()
        raw = json.loads(
            state_file.read_text(encoding="utf-8"),
        )
        assert "arms" in raw
        assert raw["total_pulls"] == 1

    def test_load_on_init(self, tmp_path: Path):
        state_file = tmp_path / "bandit_state.json"

        loop1 = FeedbackLoop(state_path=state_file)
        arm_name, _ = loop1.before_query(
            "specification", corpus_size=10,
        )
        answer = _make_answer(
            confidence=0.9, grounded=True, evidence_count=3,
        )
        loop1.after_query(arm_name, answer)

        loop2 = FeedbackLoop(state_path=state_file)
        assert loop2.selector.bandit.total_pulls == 1
        assert arm_name in loop2.selector.bandit.arms
        stats = loop2.selector.bandit.arms[arm_name]
        assert stats.pulls == 1

    def test_round_trip_preserves_all_state(
        self, tmp_path: Path,
    ):
        state_file = tmp_path / "bandit_state.json"
        loop1 = FeedbackLoop(state_path=state_file)

        for _ in range(10):
            arm_name, _ = loop1.before_query(
                "summary", corpus_size=10,
            )
            answer = _make_answer(
                confidence=0.7,
                grounded=True,
                evidence_count=2,
            )
            loop1.after_query(arm_name, answer)

        loop2 = FeedbackLoop(state_path=state_file)
        assert loop2.selector.bandit.total_pulls == 10
        for name in loop1.selector.bandit.arms:
            orig = loop1.selector.bandit.arms[name]
            rest = loop2.selector.bandit.arms[name]
            assert orig.pulls == rest.pulls
            assert abs(
                orig.mean_reward - rest.mean_reward,
            ) < 1e-12

    def test_no_state_file_starts_fresh(self, tmp_path: Path):
        state_file = tmp_path / "does_not_exist.json"
        loop = FeedbackLoop(state_path=state_file)
        assert loop.selector.bandit.total_pulls == 0
        assert not loop.selector._cold_started

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        state_file = tmp_path / "nested" / "dir" / "state.json"
        loop = FeedbackLoop(state_path=state_file)
        arm_name, _ = loop.before_query(
            "specification", corpus_size=10,
        )
        answer = _make_answer(
            confidence=0.5, grounded=False, evidence_count=1,
        )
        loop.after_query(arm_name, answer)

        assert state_file.exists()


# -------------------------------------------------------------------
# 5. Integration: Full RL loop with real RAGAnswer
# -------------------------------------------------------------------


class TestFeedbackLoopIntegration:
    """Full integration: real RAGAnswer through feedback loop."""

    def test_full_loop_with_real_answer(self):
        """Complete cycle: select, build answer, reward, update."""
        loop = FeedbackLoop()
        registry = _build_default_registry()

        arm_name, techniques = loop.before_query(
            "specification",
            corpus_size=10,
            registry=registry,
        )
        assert arm_name in DEFAULT_ARMS
        assert isinstance(techniques, set)

        evidence = _make_evidence(n=3)
        citations = _make_citations(evidence)
        trace = _make_trace(overall=0.85, grounded=True)
        answer = RAGAnswer(
            answer="Triage uses vital signs for acuity scoring.",
            evidence=evidence,
            citations=citations,
            confidence=0.85,
            grounded=True,
            trace=trace,
        )

        reward = loop.after_query(arm_name, answer)
        assert 0.0 <= reward <= 1.0
        assert loop.selector.bandit.total_pulls == 1

    def test_bandit_learns_from_answer_quality(self):
        """Bandit prefers arms yielding higher-quality answers."""
        loop = FeedbackLoop()

        loop.selector.register_arm(
            "low_quality",
            {"sparse_bm25", "confidence_scoring"},
        )
        loop.selector.register_arm(
            "mid_quality",
            {"sparse_bm25", "dense_vector", "rrf_fusion"},
        )
        loop.selector.register_arm(
            "high_quality",
            {
                "sparse_bm25",
                "dense_vector",
                "rrf_fusion",
                "query_expansion",
            },
        )

        quality_map = {
            "low_quality": (0.2, False, 1),
            "mid_quality": (0.5, True, 2),
            "high_quality": (0.9, True, 3),
        }

        for _ in range(100):
            arm_name, _ = loop.before_query(
                "summary", corpus_size=10,
            )
            conf, grounded, ev_count = quality_map[arm_name]
            answer = _make_answer(
                confidence=conf,
                grounded=grounded,
                evidence_count=ev_count,
            )
            loop.after_query(arm_name, answer)

        assert loop.selector.bandit.best_arm() == "high_quality"

    def test_compose_pipeline_with_bandit_techniques(self):
        """Technique set works as enabled_overrides."""
        loop = FeedbackLoop()
        registry = _build_default_registry()

        _arm_name, techniques = loop.before_query(
            "summary", corpus_size=10, registry=registry,
        )

        plan = compose_pipeline(
            registry,
            "summary",
            corpus_size=10,
            enabled_overrides=techniques,
        )
        assert len(plan.steps) > 0
        assert plan.query_type == "summary"

    def test_persistence_across_sessions(
        self, tmp_path: Path,
    ):
        """State survives across FeedbackLoop instances."""
        state_file = tmp_path / "session_state.json"
        registry = _build_default_registry()

        loop1 = FeedbackLoop(state_path=state_file)
        for _ in range(5):
            arm_name, _ = loop1.before_query(
                "specification",
                corpus_size=10,
                registry=registry,
            )
            answer = _make_answer(
                confidence=0.8,
                grounded=True,
                evidence_count=3,
            )
            loop1.after_query(arm_name, answer)

        loop2 = FeedbackLoop(state_path=state_file)
        assert loop2.selector.bandit.total_pulls == 5

        arm_name, _techniques = loop2.before_query(
            "specification",
            corpus_size=10,
            registry=registry,
        )
        answer = _make_answer(
            confidence=0.9, grounded=True, evidence_count=3,
        )
        loop2.after_query(arm_name, answer)
        assert loop2.selector.bandit.total_pulls == 6


# -------------------------------------------------------------------
# 6. Exports
# -------------------------------------------------------------------


class TestExports:
    """Verify public API exports from cam_rag.rl."""

    def test_feedback_types_exported(self):
        from cam_rag.rl import (  # noqa: F401
            BanditSelector,
            FeedbackLoop,
            RewardComputer,
        )

    def test_default_arms_exported(self):
        from cam_rag.rl import DEFAULT_ARMS

        assert "base_retrieval" in DEFAULT_ARMS
        assert "base+expansion" in DEFAULT_ARMS
        assert "full_pipeline" in DEFAULT_ARMS
