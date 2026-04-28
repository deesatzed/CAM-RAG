"""UCB1 multi-armed bandit for technique selection.

Each arm represents a retrieval pipeline configuration. Rewards are
retrieval quality scores in [0.0, 1.0] computed from evaluation metrics
(recall@k, term match, confidence).  The bandit balances exploration
of new technique combinations against exploitation of known good ones.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class ArmStats:
    """Running statistics for one bandit arm (technique combination)."""

    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.pulls


@dataclass(slots=True)
class UCB1Bandit:
    """Upper Confidence Bound bandit for technique selection.

    Each arm represents a technique combination (pipeline configuration).
    Rewards are retrieval quality scores in [0.0, 1.0].
    """

    arms: dict[str, ArmStats] = field(default_factory=dict)
    total_pulls: int = 0
    exploration_weight: float = 2.0

    def register_arm(self, name: str) -> None:
        """Add a new arm if it does not already exist."""
        if name not in self.arms:
            self.arms[name] = ArmStats()

    def select(self) -> str:
        """Select the arm with the highest UCB1 score."""
        if not self.arms:
            raise ValueError("No arms registered")

        # Cold start: try each arm once
        for name, stats in self.arms.items():
            if stats.pulls == 0:
                return name

        best_name = ""
        best_score = -1.0
        for name, stats in self.arms.items():
            ucb = stats.mean_reward + self.exploration_weight * math.sqrt(
                math.log(self.total_pulls) / stats.pulls
            )
            if ucb > best_score:
                best_score = ucb
                best_name = name
        return best_name

    def update(self, name: str, reward: float) -> None:
        """Record a reward for an arm pull."""
        if name not in self.arms:
            raise KeyError(f"Unknown arm: {name}")
        if not (0.0 <= reward <= 1.0):
            raise ValueError(f"Reward must be in [0, 1], got {reward}")
        self.arms[name].update(reward)
        self.total_pulls += 1

    @property
    def arm_names(self) -> set[str]:
        """Set of all registered arm names."""
        return set(self.arms)

    def best_arm(self) -> str:
        """Return the arm with the highest observed mean reward."""
        if not self.arms:
            raise ValueError("No arms registered")
        return max(self.arms, key=lambda n: self.arms[n].mean_reward)


@dataclass(slots=True)
class ABTestResult:
    """Result of an A/B comparison between two technique pipelines."""

    arm_a: str
    arm_b: str
    a_mean: float
    b_mean: float
    a_pulls: int
    b_pulls: int
    winner: str
    lift: float  # (winner_mean - loser_mean) / loser_mean


def run_ab_test(
    bandit: UCB1Bandit,
    arm_a: str,
    arm_b: str,
    reward_fn: dict[str, float],
    n_rounds: int,
    *,
    noise: float = 0.03,
) -> ABTestResult:
    """Run an A/B test between two specific arms using the bandit."""
    rng = random.Random(7)
    for _ in range(n_rounds):
        arm = bandit.select()
        if arm not in (arm_a, arm_b):
            arm = rng.choice([arm_a, arm_b])
        base = reward_fn.get(arm, 0.5)
        reward = max(0.0, min(1.0, base + rng.gauss(0, noise)))
        bandit.update(arm, reward)

    stats_a = bandit.arms[arm_a]
    stats_b = bandit.arms[arm_b]
    winner = arm_a if stats_a.mean_reward >= stats_b.mean_reward else arm_b
    loser_mean = min(stats_a.mean_reward, stats_b.mean_reward)
    winner_mean = max(stats_a.mean_reward, stats_b.mean_reward)
    lift = (winner_mean - loser_mean) / loser_mean if loser_mean > 0 else float("inf")

    return ABTestResult(
        arm_a=arm_a,
        arm_b=arm_b,
        a_mean=stats_a.mean_reward,
        b_mean=stats_b.mean_reward,
        a_pulls=stats_a.pulls,
        b_pulls=stats_b.pulls,
        winner=winner,
        lift=lift,
    )
