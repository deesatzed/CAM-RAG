"""Tests for technique fitness scoring and lifecycle governance.

Validates FitnessScore weighted sums, FitnessTracker EMA convergence,
LifecycleManager state transitions, and their integration.

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

import pytest

from cam_rag.pipeline.governance import (
    ACTIVE_TO_PREFERRED_FITNESS,
    DECLINING_TO_DISABLED_CONSECUTIVE,
    DECLINING_TO_DISABLED_FITNESS,
    DEFAULT_EMA_ALPHA,
    EMBRYONIC_OBSERVATION_MIN,
    PREFERRED_TO_DECLINING_FITNESS,
    W_CONFIDENCE_LIFT,
    W_DIVERSITY_CONTRIBUTION,
    W_LATENCY_EFFICIENCY,
    W_QUERY_TYPE_AFFINITY,
    W_RECENCY,
    W_RETRIEVAL_QUALITY,
    FitnessScore,
    FitnessTracker,
    LifecycleManager,
    TechniqueState,
)

# ===================================================================
# 1. FitnessScore
# ===================================================================


class TestFitnessScoreWeightedSum:
    """FitnessScore.overall must be the correct weighted sum."""

    def test_all_ones_equals_one(self):
        score = FitnessScore(
            retrieval_quality=1.0,
            query_type_affinity=1.0,
            confidence_lift=1.0,
            diversity_contribution=1.0,
            latency_efficiency=1.0,
            recency=1.0,
        )
        assert score.overall == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros_equals_zero(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(0.0, abs=1e-9)

    def test_single_dimension_retrieval_quality(self):
        score = FitnessScore(
            retrieval_quality=1.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(W_RETRIEVAL_QUALITY, abs=1e-9)

    def test_single_dimension_query_type_affinity(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=1.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(
            W_QUERY_TYPE_AFFINITY, abs=1e-9
        )

    def test_single_dimension_confidence_lift(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=1.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(
            W_CONFIDENCE_LIFT, abs=1e-9
        )

    def test_single_dimension_diversity_contribution(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=1.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(
            W_DIVERSITY_CONTRIBUTION, abs=1e-9
        )

    def test_single_dimension_latency_efficiency(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=1.0,
            recency=0.0,
        )
        assert score.overall == pytest.approx(
            W_LATENCY_EFFICIENCY, abs=1e-9
        )

    def test_single_dimension_recency(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=1.0,
        )
        assert score.overall == pytest.approx(W_RECENCY, abs=1e-9)

    def test_all_dimensions_contribute(self):
        """Each dimension adds a non-zero amount to the overall."""
        base = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        dimensions = [
            "retrieval_quality",
            "query_type_affinity",
            "confidence_lift",
            "diversity_contribution",
            "latency_efficiency",
            "recency",
        ]
        for dim in dimensions:
            kwargs = dict.fromkeys(dimensions, 0.0)
            kwargs[dim] = 0.5
            score = FitnessScore(**kwargs)
            assert score.overall > base.overall, (
                f"{dim} did not increase overall"
            )

    def test_weights_sum_to_one(self):
        total_weight = (
            W_RETRIEVAL_QUALITY
            + W_QUERY_TYPE_AFFINITY
            + W_CONFIDENCE_LIFT
            + W_DIVERSITY_CONTRIBUTION
            + W_LATENCY_EFFICIENCY
            + W_RECENCY
        )
        assert total_weight == pytest.approx(1.0, abs=1e-9)

    def test_mixed_values(self):
        score = FitnessScore(
            retrieval_quality=0.8,
            query_type_affinity=0.6,
            confidence_lift=0.5,
            diversity_contribution=0.7,
            latency_efficiency=0.9,
            recency=0.3,
        )
        expected = (
            0.30 * 0.8
            + 0.20 * 0.6
            + 0.15 * 0.5
            + 0.15 * 0.7
            + 0.10 * 0.9
            + 0.10 * 0.3
        )
        assert score.overall == pytest.approx(expected, abs=1e-9)


class TestFitnessScoreBoundaries:
    """FitnessScore must enforce [0.0, 1.0] range on all dimensions."""

    def test_boundary_zero(self):
        score = FitnessScore(
            retrieval_quality=0.0,
            query_type_affinity=0.0,
            confidence_lift=0.0,
            diversity_contribution=0.0,
            latency_efficiency=0.0,
            recency=0.0,
        )
        assert score.overall == 0.0

    def test_boundary_one(self):
        score = FitnessScore(
            retrieval_quality=1.0,
            query_type_affinity=1.0,
            confidence_lift=1.0,
            diversity_contribution=1.0,
            latency_efficiency=1.0,
            recency=1.0,
        )
        assert score.overall == pytest.approx(1.0, abs=1e-9)

    def test_rejects_negative_value(self):
        with pytest.raises(ValueError, match="must be in"):
            FitnessScore(
                retrieval_quality=-0.1,
                query_type_affinity=0.5,
                confidence_lift=0.5,
                diversity_contribution=0.5,
                latency_efficiency=0.5,
                recency=0.5,
            )

    def test_rejects_value_above_one(self):
        with pytest.raises(ValueError, match="must be in"):
            FitnessScore(
                retrieval_quality=0.5,
                query_type_affinity=1.1,
                confidence_lift=0.5,
                diversity_contribution=0.5,
                latency_efficiency=0.5,
                recency=0.5,
            )

    def test_rejects_each_dimension_out_of_range(self):
        dimensions = [
            "retrieval_quality",
            "query_type_affinity",
            "confidence_lift",
            "diversity_contribution",
            "latency_efficiency",
            "recency",
        ]
        for dim in dimensions:
            kwargs = dict.fromkeys(dimensions, 0.5)
            kwargs[dim] = 1.5
            with pytest.raises(ValueError, match="must be in"):
                FitnessScore(**kwargs)

    def test_frozen_prevents_mutation(self):
        score = FitnessScore(
            retrieval_quality=0.5,
            query_type_affinity=0.5,
            confidence_lift=0.5,
            diversity_contribution=0.5,
            latency_efficiency=0.5,
            recency=0.5,
        )
        with pytest.raises(AttributeError):
            score.retrieval_quality = 0.9  # type: ignore[misc]


# ===================================================================
# 2. FitnessTracker
# ===================================================================


class TestFitnessTrackerEMA:
    """FitnessTracker EMA must converge toward repeated observations."""

    @staticmethod
    def _make_score(overall_approx: float) -> FitnessScore:
        """Build a FitnessScore where all dimensions equal the target."""
        v = max(0.0, min(1.0, overall_approx))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_first_observation_seeds_ema(self):
        tracker = FitnessTracker()
        score = self._make_score(0.8)
        tracker.record("dense_only", score)
        assert tracker.get_fitness("dense_only") == pytest.approx(
            score.overall, abs=1e-9
        )

    def test_ema_converges_toward_constant_signal(self):
        tracker = FitnessTracker(alpha=0.3)
        target = 0.9
        score = self._make_score(target)
        for _ in range(50):
            tracker.record("technique_a", score)
        assert tracker.get_fitness("technique_a") == pytest.approx(
            score.overall, abs=1e-4
        )

    def test_ema_adapts_when_signal_changes(self):
        tracker = FitnessTracker(alpha=0.3)
        high = self._make_score(0.9)
        low = self._make_score(0.2)

        for _ in range(30):
            tracker.record("t", high)
        fitness_after_high = tracker.get_fitness("t")

        for _ in range(30):
            tracker.record("t", low)
        fitness_after_low = tracker.get_fitness("t")

        assert fitness_after_low < fitness_after_high

    def test_higher_alpha_adapts_faster(self):
        slow = FitnessTracker(alpha=0.1)
        fast = FitnessTracker(alpha=0.9)

        high = self._make_score(0.9)
        low = self._make_score(0.1)

        # Seed both with high
        for _ in range(10):
            slow.record("t", high)
            fast.record("t", high)

        # Shift to low
        for _ in range(5):
            slow.record("t", low)
            fast.record("t", low)

        # Fast tracker should be closer to low.overall
        assert fast.get_fitness("t") < slow.get_fitness("t")

    def test_unknown_technique_returns_zero(self):
        tracker = FitnessTracker()
        assert tracker.get_fitness("nonexistent") == 0.0


class TestFitnessTrackerMultipleTechniques:
    """FitnessTracker must track techniques independently."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_independent_tracking(self):
        tracker = FitnessTracker()
        high = self._make_score(0.9)
        low = self._make_score(0.1)

        for _ in range(20):
            tracker.record("good", high)
            tracker.record("bad", low)

        assert tracker.get_fitness("good") > tracker.get_fitness("bad")
        assert tracker.get_fitness("good") > 0.8
        assert tracker.get_fitness("bad") < 0.2

    def test_ranking_order(self):
        tracker = FitnessTracker()
        scores = {"alpha": 0.9, "beta": 0.5, "gamma": 0.1}
        for _ in range(30):
            for name, val in scores.items():
                tracker.record(name, self._make_score(val))

        ranked = tracker.rank_techniques()
        names = [name for name, _ in ranked]
        assert names == ["alpha", "beta", "gamma"]

    def test_observation_count_increments(self):
        tracker = FitnessTracker()
        score = self._make_score(0.5)
        assert tracker.observation_count("t") == 0
        tracker.record("t", score)
        assert tracker.observation_count("t") == 1
        tracker.record("t", score)
        assert tracker.observation_count("t") == 2

    def test_technique_names_property(self):
        tracker = FitnessTracker()
        score = self._make_score(0.5)
        tracker.record("a", score)
        tracker.record("b", score)
        assert tracker.technique_names == {"a", "b"}


# ===================================================================
# 3. LifecycleManager -- State Machine Transitions
# ===================================================================


class TestLifecycleEmbryonicToActive:
    """embryonic -> active after enough observations."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_stays_embryonic_below_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        score = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN - 1):
            tracker.record("t", score)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.EMBRYONIC

    def test_transitions_to_active_at_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        score = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("t", score)
        transitions = mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.ACTIVE
        assert "t" in transitions
        assert transitions["t"] == (
            TechniqueState.EMBRYONIC,
            TechniqueState.ACTIVE,
        )

    def test_default_state_is_embryonic(self):
        mgr = LifecycleManager()
        assert mgr.get_state("unknown") == TechniqueState.EMBRYONIC


class TestLifecycleActiveToPreferred:
    """active -> preferred when fitness > threshold."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_stays_active_below_preferred_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        # Build up enough observations to reach active
        score = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("t", score)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.ACTIVE

        # Fitness at 0.5 is below the 0.7 preferred threshold
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.ACTIVE

    def test_transitions_to_preferred_above_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        # Reach active state first
        score_mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("t", score_mid)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.ACTIVE

        # Push fitness above preferred threshold
        score_high = self._make_score(1.0)
        for _ in range(30):
            tracker.record("t", score_high)
        fitness = tracker.get_fitness("t")
        assert fitness > ACTIVE_TO_PREFERRED_FITNESS

        transitions = mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.PREFERRED
        assert "t" in transitions


class TestLifecyclePreferredToDeclining:
    """preferred -> declining when fitness drops below threshold."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def _bring_to_preferred(
        self, tracker: FitnessTracker, mgr: LifecycleManager, name: str
    ) -> None:
        """Helper to advance a technique to PREFERRED state."""
        score_mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record(name, score_mid)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.ACTIVE

        score_high = self._make_score(1.0)
        for _ in range(30):
            tracker.record(name, score_high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.PREFERRED

    def test_transitions_to_declining_below_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        self._bring_to_preferred(tracker, mgr, "t")

        # Crush fitness below declining threshold
        score_low = self._make_score(0.0)
        for _ in range(100):
            tracker.record("t", score_low)
        assert tracker.get_fitness("t") < PREFERRED_TO_DECLINING_FITNESS

        transitions = mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DECLINING
        assert "t" in transitions

    def test_stays_preferred_above_declining_threshold(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        self._bring_to_preferred(tracker, mgr, "t")

        # Keep fitness above declining threshold
        score_ok = self._make_score(0.8)
        for _ in range(20):
            tracker.record("t", score_ok)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.PREFERRED


class TestLifecycleDecliningToDisabled:
    """declining -> disabled after consecutive low-fitness checks."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def _bring_to_declining(
        self, tracker: FitnessTracker, mgr: LifecycleManager, name: str
    ) -> None:
        """Helper to advance a technique to DECLINING state."""
        score_mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record(name, score_mid)
        mgr.evaluate_transitions(tracker)

        score_high = self._make_score(1.0)
        for _ in range(30):
            tracker.record(name, score_high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.PREFERRED

        score_low = self._make_score(0.0)
        for _ in range(100):
            tracker.record(name, score_low)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.DECLINING

    def test_needs_consecutive_checks_to_disable(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        self._bring_to_declining(tracker, mgr, "t")

        # Fitness is already very low -- keep it there
        score_low = self._make_score(0.0)
        for _ in range(10):
            tracker.record("t", score_low)

        # Need DECLINING_TO_DISABLED_CONSECUTIVE evaluate calls
        for i in range(DECLINING_TO_DISABLED_CONSECUTIVE - 1):
            mgr.evaluate_transitions(tracker)
            assert mgr.get_state("t") == TechniqueState.DECLINING, (
                f"Disabled too early at check {i + 1}"
            )

        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DISABLED

    def test_streak_resets_if_fitness_recovers(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        self._bring_to_declining(tracker, mgr, "t")

        # Accumulate some declining checks
        score_low = self._make_score(0.0)
        for _ in range(10):
            tracker.record("t", score_low)
        for _ in range(DECLINING_TO_DISABLED_CONSECUTIVE - 2):
            mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DECLINING

        # Push fitness above the disabled threshold (but still below
        # declining threshold so it stays declining)
        score_mid = self._make_score(0.5)
        for _ in range(60):
            tracker.record("t", score_mid)
        # This should reset the streak
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DECLINING

        # Now needs the full consecutive count again
        score_low2 = self._make_score(0.0)
        for _ in range(100):
            tracker.record("t", score_low2)
        for _ in range(DECLINING_TO_DISABLED_CONSECUTIVE - 1):
            mgr.evaluate_transitions(tracker)
            assert mgr.get_state("t") == TechniqueState.DECLINING

        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DISABLED


class TestLifecycleReactivation:
    """Disabled techniques can be manually reactivated."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_reactivate_sets_embryonic(self):
        mgr = LifecycleManager()
        mgr._states["t"] = TechniqueState.DISABLED
        mgr.reactivate("t")
        assert mgr.get_state("t") == TechniqueState.EMBRYONIC

    def test_reactivate_clears_declining_streak(self):
        mgr = LifecycleManager()
        mgr._states["t"] = TechniqueState.DISABLED
        mgr._declining_streak["t"] = 10
        mgr.reactivate("t")
        assert mgr._declining_streak.get("t") is None

    def test_reactivate_unknown_technique(self):
        """Reactivating a non-existent technique sets it to embryonic."""
        mgr = LifecycleManager()
        mgr.reactivate("new_tech")
        assert mgr.get_state("new_tech") == TechniqueState.EMBRYONIC

    def test_disabled_does_not_transition_without_reactivation(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        mgr._states["t"] = TechniqueState.DISABLED

        # Even with observations the disabled state sticks
        score_high = self._make_score(1.0)
        for _ in range(30):
            tracker.record("t", score_high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("t") == TechniqueState.DISABLED


# ===================================================================
# 4. Full State Machine Traversal
# ===================================================================


class TestFullLifecycleTraversal:
    """End-to-end: embryonic -> active -> preferred -> declining -> disabled."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_complete_lifecycle(self):
        tracker = FitnessTracker()
        mgr = LifecycleManager()
        name = "hybrid_rrf"

        # -- embryonic --
        assert mgr.get_state(name) == TechniqueState.EMBRYONIC

        # Accumulate observations to transition to active
        score_mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record(name, score_mid)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.ACTIVE

        # -- active -> preferred --
        score_high = self._make_score(1.0)
        for _ in range(30):
            tracker.record(name, score_high)
        assert tracker.get_fitness(name) > ACTIVE_TO_PREFERRED_FITNESS
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.PREFERRED

        # -- preferred -> declining --
        score_low = self._make_score(0.0)
        for _ in range(100):
            tracker.record(name, score_low)
        assert tracker.get_fitness(name) < PREFERRED_TO_DECLINING_FITNESS
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.DECLINING

        # -- declining -> disabled --
        for _ in range(50):
            tracker.record(name, score_low)
        assert tracker.get_fitness(name) < DECLINING_TO_DISABLED_FITNESS
        for _ in range(DECLINING_TO_DISABLED_CONSECUTIVE):
            mgr.evaluate_transitions(tracker)
        assert mgr.get_state(name) == TechniqueState.DISABLED


# ===================================================================
# 5. Integration: Tracker + Lifecycle Working Together
# ===================================================================


class TestTrackerLifecycleIntegration:
    """FitnessTracker and LifecycleManager work together end-to-end."""

    @staticmethod
    def _make_score(val: float) -> FitnessScore:
        v = max(0.0, min(1.0, val))
        return FitnessScore(
            retrieval_quality=v,
            query_type_affinity=v,
            confidence_lift=v,
            diversity_contribution=v,
            latency_efficiency=v,
            recency=v,
        )

    def test_multiple_techniques_different_fates(self):
        """Two techniques: one thrives, one declines."""
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        good = self._make_score(0.95)
        bad = self._make_score(0.05)

        # Both reach active
        mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("winner", mid)
            tracker.record("loser", mid)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("winner") == TechniqueState.ACTIVE
        assert mgr.get_state("loser") == TechniqueState.ACTIVE

        # Winner gets high scores, loser gets low scores
        for _ in range(40):
            tracker.record("winner", good)
            tracker.record("loser", bad)
        mgr.evaluate_transitions(tracker)

        # Winner should be preferred
        assert mgr.get_state("winner") == TechniqueState.PREFERRED
        # Loser stays active (fitness still above declining threshold
        # since EMA hasn't dropped far enough yet from the 0.5 seed)
        # or is still active
        loser_state = mgr.get_state("loser")
        assert loser_state in (
            TechniqueState.ACTIVE,
            TechniqueState.PREFERRED,
        )

        # Rankings should reflect the fitness difference
        ranked = tracker.rank_techniques()
        assert ranked[0][0] == "winner"
        assert ranked[-1][0] == "loser"

    def test_ranking_reflects_lifecycle_state(self):
        """Higher-ranked techniques should generally be in higher states."""
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        scores = {"best": 1.0, "mid": 0.5, "worst": 0.1}
        for _ in range(EMBRYONIC_OBSERVATION_MIN + 30):
            for name, val in scores.items():
                tracker.record(name, self._make_score(val))
            mgr.evaluate_transitions(tracker)

        ranked_names = [n for n, _ in tracker.rank_techniques()]
        assert ranked_names[0] == "best"
        assert ranked_names[-1] == "worst"

        assert mgr.get_state("best") == TechniqueState.PREFERRED

    def test_reactivated_technique_can_re_prove(self):
        """A disabled technique can come back and earn preferred again."""
        tracker = FitnessTracker()
        mgr = LifecycleManager()

        # Fast-track to disabled
        mid = self._make_score(0.5)
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("phoenix", mid)
        mgr.evaluate_transitions(tracker)

        high = self._make_score(1.0)
        for _ in range(30):
            tracker.record("phoenix", high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("phoenix") == TechniqueState.PREFERRED

        low = self._make_score(0.0)
        for _ in range(100):
            tracker.record("phoenix", low)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("phoenix") == TechniqueState.DECLINING

        for _ in range(50):
            tracker.record("phoenix", low)
        for _ in range(DECLINING_TO_DISABLED_CONSECUTIVE):
            mgr.evaluate_transitions(tracker)
        assert mgr.get_state("phoenix") == TechniqueState.DISABLED

        # Reactivate
        mgr.reactivate("phoenix")
        assert mgr.get_state("phoenix") == TechniqueState.EMBRYONIC

        # Re-prove with new high scores
        for _ in range(EMBRYONIC_OBSERVATION_MIN):
            tracker.record("phoenix", high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("phoenix") == TechniqueState.ACTIVE

        for _ in range(30):
            tracker.record("phoenix", high)
        mgr.evaluate_transitions(tracker)
        assert mgr.get_state("phoenix") == TechniqueState.PREFERRED


# ===================================================================
# 6. TechniqueState Enum
# ===================================================================


class TestTechniqueStateEnum:
    """TechniqueState enum has the expected members and values."""

    def test_all_states_exist(self):
        expected = {
            "EMBRYONIC", "ACTIVE", "PREFERRED", "DECLINING", "DISABLED",
        }
        actual = {s.name for s in TechniqueState}
        assert actual == expected

    def test_state_values(self):
        assert TechniqueState.EMBRYONIC.value == "embryonic"
        assert TechniqueState.ACTIVE.value == "active"
        assert TechniqueState.PREFERRED.value == "preferred"
        assert TechniqueState.DECLINING.value == "declining"
        assert TechniqueState.DISABLED.value == "disabled"


# ===================================================================
# 7. Module-level constants
# ===================================================================


class TestModuleConstants:
    """Module constants have expected default values."""

    def test_default_ema_alpha(self):
        assert DEFAULT_EMA_ALPHA == 0.3

    def test_embryonic_observation_min(self):
        assert EMBRYONIC_OBSERVATION_MIN == 10

    def test_preferred_threshold(self):
        assert ACTIVE_TO_PREFERRED_FITNESS == 0.7

    def test_declining_threshold(self):
        assert PREFERRED_TO_DECLINING_FITNESS == 0.4

    def test_disabled_threshold(self):
        assert DECLINING_TO_DISABLED_FITNESS == 0.2

    def test_disabled_consecutive(self):
        assert DECLINING_TO_DISABLED_CONSECUTIVE == 5


# ===================================================================
# 8. Package-level imports
# ===================================================================


class TestPackageExports:
    """The pipeline package re-exports governance types."""

    def test_fitness_score_importable(self):
        from cam_rag.pipeline import FitnessScore as FitnessScorePkg
        assert FitnessScorePkg is FitnessScore

    def test_fitness_tracker_importable(self):
        from cam_rag.pipeline import FitnessTracker as FitnessTrackerPkg
        assert FitnessTrackerPkg is FitnessTracker

    def test_technique_state_importable(self):
        from cam_rag.pipeline import TechniqueState as TechniqueStatePkg
        assert TechniqueStatePkg is TechniqueState

    def test_lifecycle_manager_importable(self):
        from cam_rag.pipeline import LifecycleManager as LifecycleManagerPkg
        assert LifecycleManagerPkg is LifecycleManager
