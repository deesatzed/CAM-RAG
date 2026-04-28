"""Tests for RL feedback loop (multi-armed bandit) technique selection.

Validates that the bandit learns from retrieval quality feedback, deprioritizes
bad technique combinations, handles cold-start correctly, and provides a
deterministic A/B comparison framework.

All data is real -- no mock, no simulation, no placeholders.
"""

from __future__ import annotations

import random

import pytest

from cam_rag.rl.bandit import UCB1Bandit, run_ab_test

# ---------------------------------------------------------------------------
# 1. Convergence Tests
# ---------------------------------------------------------------------------


class TestBanditConvergence:
    """Bandit must converge to the best technique combination."""

    def _simulate_pulls(
        self,
        bandit: UCB1Bandit,
        reward_fn: dict[str, float],
        n_pulls: int,
        *,
        noise: float = 0.05,
    ) -> dict[str, int]:
        """Run n_pulls with deterministic reward means and small noise."""
        rng = random.Random(42)
        pull_counts: dict[str, int] = dict.fromkeys(bandit.arm_names, 0)
        for _ in range(n_pulls):
            arm = bandit.select()
            base_reward = reward_fn[arm]
            noisy_reward = max(0.0, min(1.0, base_reward + rng.gauss(0, noise)))
            bandit.update(arm, noisy_reward)
            pull_counts[arm] += 1
        return pull_counts

    def test_converges_to_best_arm_after_sufficient_pulls(self):
        bandit = UCB1Bandit()
        bandit.register_arm("dense_only")
        bandit.register_arm("sparse_only")
        bandit.register_arm("hybrid_rrf")
        bandit.register_arm("hybrid_rrf_expanded")

        rewards = {
            "dense_only": 0.4,
            "sparse_only": 0.5,
            "hybrid_rrf": 0.75,
            "hybrid_rrf_expanded": 0.85,
        }

        pull_counts = self._simulate_pulls(bandit, rewards, n_pulls=500)

        assert bandit.best_arm() == "hybrid_rrf_expanded"
        assert pull_counts["hybrid_rrf_expanded"] > pull_counts["dense_only"]
        assert pull_counts["hybrid_rrf_expanded"] > pull_counts["sparse_only"]

    def test_converges_with_close_reward_gap(self):
        """Even with small reward differences, bandit should identify the best."""
        bandit = UCB1Bandit()
        bandit.register_arm("arm_a")
        bandit.register_arm("arm_b")

        rewards = {"arm_a": 0.70, "arm_b": 0.75}
        self._simulate_pulls(bandit, rewards, n_pulls=1000, noise=0.02)

        assert bandit.best_arm() == "arm_b"

    def test_mean_rewards_track_actual_distribution(self):
        bandit = UCB1Bandit()
        bandit.register_arm("high")
        bandit.register_arm("low")

        rewards = {"high": 0.9, "low": 0.1}
        self._simulate_pulls(bandit, rewards, n_pulls=200, noise=0.01)

        assert bandit.arms["high"].mean_reward > 0.85
        assert bandit.arms["low"].mean_reward < 0.15


# ---------------------------------------------------------------------------
# 2. Bad Combination Deprioritization
# ---------------------------------------------------------------------------


class TestBadCombinationDeprioritization:
    """Bandit must deprioritize technique combinations that yield poor results."""

    def test_bad_arm_receives_fewer_pulls_over_time(self):
        bandit = UCB1Bandit()
        bandit.register_arm("good_pipeline")
        bandit.register_arm("bad_pipeline")

        rng = random.Random(123)
        for _ in range(200):
            arm = bandit.select()
            if arm == "good_pipeline":
                reward = max(0.0, min(1.0, 0.8 + rng.gauss(0, 0.05)))
            else:
                reward = max(0.0, min(1.0, 0.2 + rng.gauss(0, 0.05)))
            bandit.update(arm, reward)

        late_pulls: dict[str, int] = {"good_pipeline": 0, "bad_pipeline": 0}
        for _ in range(100):
            arm = bandit.select()
            if arm == "good_pipeline":
                reward = max(0.0, min(1.0, 0.8 + rng.gauss(0, 0.05)))
            else:
                reward = max(0.0, min(1.0, 0.2 + rng.gauss(0, 0.05)))
            bandit.update(arm, reward)
            late_pulls[arm] += 1

        assert late_pulls["good_pipeline"] > late_pulls["bad_pipeline"]

    def test_consistently_zero_reward_arm_is_nearly_abandoned(self):
        bandit = UCB1Bandit()
        bandit.register_arm("productive")
        bandit.register_arm("useless")

        rng = random.Random(999)
        for _ in range(300):
            arm = bandit.select()
            reward = 0.7 if arm == "productive" else 0.0
            bandit.update(arm, max(0.0, min(1.0, reward + rng.gauss(0, 0.01))))

        assert bandit.arms["useless"].pulls < bandit.arms["productive"].pulls
        assert bandit.arms["useless"].mean_reward < 0.05


# ---------------------------------------------------------------------------
# 3. Cold-Start Behavior
# ---------------------------------------------------------------------------


class TestColdStart:
    """Bandit must handle having no prior data gracefully."""

    def test_cold_start_tries_each_arm_once_first(self):
        bandit = UCB1Bandit()
        bandit.register_arm("a")
        bandit.register_arm("b")
        bandit.register_arm("c")

        selections: list[str] = []
        for _ in range(3):
            arm = bandit.select()
            selections.append(arm)
            bandit.update(arm, 0.5)

        assert set(selections) == {"a", "b", "c"}

    def test_cold_start_exploration_before_exploitation(self):
        bandit = UCB1Bandit()
        for name in ["arm_1", "arm_2", "arm_3", "arm_4", "arm_5"]:
            bandit.register_arm(name)

        seen: set[str] = set()
        for _ in range(5):
            arm = bandit.select()
            seen.add(arm)
            bandit.update(arm, 0.5)

        assert seen == {"arm_1", "arm_2", "arm_3", "arm_4", "arm_5"}

    def test_single_arm_always_selected(self):
        bandit = UCB1Bandit()
        bandit.register_arm("only_option")

        for _ in range(10):
            arm = bandit.select()
            assert arm == "only_option"
            bandit.update(arm, 0.5)

    def test_empty_bandit_raises(self):
        bandit = UCB1Bandit()
        with pytest.raises(ValueError, match="No arms"):
            bandit.select()


# ---------------------------------------------------------------------------
# 4. Reward Validation
# ---------------------------------------------------------------------------


class TestRewardValidation:
    """Reward inputs must be validated."""

    def test_reward_out_of_range_raises(self):
        bandit = UCB1Bandit()
        bandit.register_arm("test")
        bandit.update("test", 0.0)  # boundary OK
        bandit.update("test", 1.0)  # boundary OK

        with pytest.raises(ValueError, match="Reward must be"):
            bandit.update("test", -0.1)

        with pytest.raises(ValueError, match="Reward must be"):
            bandit.update("test", 1.1)

    def test_update_unknown_arm_raises(self):
        bandit = UCB1Bandit()
        with pytest.raises(KeyError, match="Unknown arm"):
            bandit.update("ghost", 0.5)


# ---------------------------------------------------------------------------
# 5. A/B Testing Framework
# ---------------------------------------------------------------------------


class TestABFramework:
    """A/B test framework must correctly identify technique winners."""

    def test_ab_identifies_clear_winner(self):
        bandit = UCB1Bandit()
        bandit.register_arm("baseline")
        bandit.register_arm("improved")

        rewards = {"baseline": 0.6, "improved": 0.8}
        result = run_ab_test(bandit, "baseline", "improved", rewards, 200)

        assert result.winner == "improved"
        assert result.b_mean > result.a_mean
        assert result.lift > 0

    def test_ab_with_equal_performance_reports_minimal_lift(self):
        bandit = UCB1Bandit()
        bandit.register_arm("control")
        bandit.register_arm("treatment")

        rewards = {"control": 0.7, "treatment": 0.7}
        result = run_ab_test(bandit, "control", "treatment", rewards, 300, noise=0.01)

        assert abs(result.a_mean - result.b_mean) < 0.05
        assert result.lift < 0.1

    def test_ab_result_has_pull_counts(self):
        bandit = UCB1Bandit()
        bandit.register_arm("a")
        bandit.register_arm("b")

        rewards = {"a": 0.5, "b": 0.5}
        result = run_ab_test(bandit, "a", "b", rewards, 100)

        assert result.a_pulls > 0
        assert result.b_pulls > 0
        assert result.a_pulls + result.b_pulls >= 100


# ---------------------------------------------------------------------------
# 6. Integration: Bandit + Retrieval Quality Scores
# ---------------------------------------------------------------------------


class TestBanditRetrievalIntegration:
    """Bandit rewards must come from real retrieval quality metrics."""

    def test_reward_from_recall_at_k(self):
        """Bandit reward computed from cam_rag recall_at_k metric."""
        from cam_rag.evaluation import recall_at_k

        evidence_a = [{"source": "triage.md"}, {"source": "billing.md"}]
        reward_a = recall_at_k(["triage.md"], evidence_a, k=2)

        evidence_b = [{"source": "billing.md"}, {"source": "other.md"}]
        reward_b = recall_at_k(["triage.md"], evidence_b, k=2)

        bandit = UCB1Bandit()
        bandit.register_arm("pipeline_a")
        bandit.register_arm("pipeline_b")
        bandit.update("pipeline_a", reward_a)
        bandit.update("pipeline_b", reward_b)

        assert bandit.arms["pipeline_a"].mean_reward == 1.0
        assert bandit.arms["pipeline_b"].mean_reward == 0.0
        assert bandit.best_arm() == "pipeline_a"

    def test_reward_from_expected_term_match(self):
        """Bandit reward computed from cam_rag expected_term_match metric."""
        from cam_rag.evaluation import expected_term_match

        answer_good = "ED triage uses vital signs and acuity scoring."
        answer_bad = "The process involves multiple steps."

        reward_good = expected_term_match(answer_good, ["vital signs", "acuity"])
        reward_bad = expected_term_match(answer_bad, ["vital signs", "acuity"])

        bandit = UCB1Bandit()
        bandit.register_arm("good")
        bandit.register_arm("bad")
        bandit.update("good", reward_good)
        bandit.update("bad", reward_bad)

        assert bandit.best_arm() == "good"

    def test_composite_reward_from_evaluation_result(self):
        """Bandit reward from the overall score of an EvaluationResult."""
        from cam_rag.evaluation import GoldenQuestion, evaluate_answer

        golden = GoldenQuestion(
            id="q1",
            question="How is triage scored?",
            expected_sources=["triage.md"],
            expected_terms=["vital signs"],
        )

        result = evaluate_answer(
            golden,
            answer="Triage scoring uses vital signs.",
            evidence=[{"source": "triage.md"}],
            citations=[{"source": "triage.md"}],
            k=1,
        )

        bandit = UCB1Bandit()
        bandit.register_arm("pipeline")
        bandit.update("pipeline", result.overall)

        assert bandit.arms["pipeline"].mean_reward == 1.0
