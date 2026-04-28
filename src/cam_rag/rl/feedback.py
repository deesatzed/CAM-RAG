"""RL feedback loop connecting UCB1Bandit to PipelineExecutor.

``RewardComputer`` turns a ``RAGAnswer`` into a scalar [0, 1] reward.
``BanditSelector`` wraps the bandit + technique registry to select
technique subsets.  ``FeedbackLoop`` ties the two together with JSON
persistence for bandit state across sessions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cam_rag.pipeline.registry import TechniqueRegistry, compose_pipeline
from cam_rag.rag.models import RAGAnswer
from cam_rag.rl.bandit import ArmStats, UCB1Bandit

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Arm definitions — each maps to a technique subset
# -----------------------------------------------------------------------

#: Default arms and the technique names each enables in the registry.
DEFAULT_ARMS: dict[str, set[str]] = {
    "base_retrieval": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "confidence_scoring",
        "citation_grounding",
        "generation",
    },
    "base+expansion": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "query_expansion",
        "confidence_scoring",
        "citation_grounding",
        "generation",
    },
    "full_pipeline": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "query_expansion",
        "cross_encoder_rerank",
        "confidence_scoring",
        "citation_grounding",
        "generation",
    },
    "moe_scored": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "moe_importance_scoring",
        "selective_filter",
        "confidence_scoring",
        "citation_grounding",
        "generation",
    },
    "moe_contracted": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "moe_importance_scoring",
        "accuracy_contracts",
        "selective_filter",
        "confidence_scoring",
        "citation_grounding",
        "generation",
    },
    "full_pseudorag": {
        "sparse_bm25",
        "dense_vector",
        "rrf_fusion",
        "moe_importance_scoring",
        "accuracy_contracts",
        "selective_filter",
        "query_expansion",
        "cross_encoder_rerank",
        "confidence_scoring",
        "citation_grounding",
        "generation",
        "etf_verification",
    },
}


# -----------------------------------------------------------------------
# RewardComputer
# -----------------------------------------------------------------------


@dataclass(slots=True)
class RewardComputer:
    """Compute a bandit reward in [0.0, 1.0] from a ``RAGAnswer``.

    The reward is a weighted combination of:
    * **confidence** — the overall confidence score from the answer.
    * **grounding** — 1.0 if grounded, 0.0 otherwise.
    * **evidence_ratio** — evidence count relative to a requested limit,
      saturating at 1.0 when ``len(evidence) >= evidence_target``.

    Weights default to (0.5, 0.3, 0.2) and can be overridden.
    """

    confidence_weight: float = 0.5
    grounding_weight: float = 0.3
    evidence_weight: float = 0.2
    evidence_target: int = 3

    def compute(self, answer: RAGAnswer) -> float:
        """Return a reward in [0.0, 1.0] for *answer*."""
        confidence = self._confidence_signal(answer)
        grounding = self._grounding_signal(answer)
        evidence_ratio = self._evidence_signal(answer)

        raw = (
            self.confidence_weight * confidence
            + self.grounding_weight * grounding
            + self.evidence_weight * evidence_ratio
        )
        return max(0.0, min(1.0, raw))

    # -- signal helpers --------------------------------------------------

    @staticmethod
    def _confidence_signal(answer: RAGAnswer) -> float:
        """Extract confidence from trace or top-level field."""
        details = answer.trace.confidence_details
        if details and "overall" in details:
            val = details["overall"]
            if isinstance(val, (int, float)):
                return max(0.0, min(1.0, float(val)))
        return max(0.0, min(1.0, answer.confidence))

    @staticmethod
    def _grounding_signal(answer: RAGAnswer) -> float:
        """Return 1.0 if the answer is grounded, else 0.0."""
        return 1.0 if answer.grounded else 0.0

    def _evidence_signal(self, answer: RAGAnswer) -> float:
        """Return evidence count ratio capped at 1.0."""
        if self.evidence_target <= 0:
            return 1.0 if answer.evidence else 0.0
        return min(1.0, len(answer.evidence) / self.evidence_target)


# -----------------------------------------------------------------------
# BanditSelector
# -----------------------------------------------------------------------


@dataclass(slots=True)
class BanditSelector:
    """Wraps ``UCB1Bandit`` + ``TechniqueRegistry`` to select techniques.

    Arms are named technique subsets (e.g. ``"base_retrieval"``).
    On first use, default arms are registered automatically.
    """

    bandit: UCB1Bandit = field(default_factory=UCB1Bandit)
    arm_techniques: dict[str, set[str]] = field(
        default_factory=dict,
    )
    _cold_started: bool = False

    def _ensure_cold_start(self) -> None:
        """Register default arms on first use if the bandit is empty."""
        if self._cold_started:
            return
        if not self.bandit.arms:
            for arm_name, techniques in DEFAULT_ARMS.items():
                self.bandit.register_arm(arm_name)
                self.arm_techniques[arm_name] = set(techniques)
        self._cold_started = True

    def register_arm(
        self, name: str, techniques: set[str],
    ) -> None:
        """Register a custom arm with its technique set."""
        self.bandit.register_arm(name)
        self.arm_techniques[name] = set(techniques)

    def select_techniques(
        self,
        query_type: str,
        corpus_size: int,
        registry: TechniqueRegistry | None = None,
    ) -> tuple[str, set[str]]:
        """Select an arm and return (arm_name, technique_set).

        Parameters
        ----------
        query_type:
            One of ``"specification"``, ``"summary"``, ``"logic"``,
            ``"synthesis"``.
        corpus_size:
            Number of chunks in the corpus (used by
            ``compose_pipeline`` filtering).
        registry:
            Optional technique registry for validation.  When
            provided, the returned set is intersected with techniques
            valid for *query_type* and *corpus_size*.
        """
        self._ensure_cold_start()
        arm_name = self.bandit.select()
        techniques = set(self.arm_techniques.get(arm_name, set()))

        if registry is not None:
            plan = compose_pipeline(
                registry,
                query_type,
                corpus_size=corpus_size,
                enabled_overrides=techniques,
            )
            techniques = set(plan.steps)

        return arm_name, techniques

    def record_reward(self, arm_name: str, reward: float) -> None:
        """Record a reward for the given arm."""
        self.bandit.update(arm_name, reward)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise bandit + arm technique mapping to a plain dict."""
        arms_data: dict[str, dict[str, Any]] = {}
        for name, stats in self.bandit.arms.items():
            arms_data[name] = {
                "pulls": stats.pulls,
                "total_reward": stats.total_reward,
                "mean_reward": stats.mean_reward,
                "techniques": sorted(
                    self.arm_techniques.get(name, set()),
                ),
            }
        return {
            "total_pulls": self.bandit.total_pulls,
            "exploration_weight": self.bandit.exploration_weight,
            "arms": arms_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BanditSelector:
        """Restore a ``BanditSelector`` from a serialised dict."""
        bandit = UCB1Bandit(
            total_pulls=data.get("total_pulls", 0),
            exploration_weight=data.get("exploration_weight", 2.0),
        )
        arm_techniques: dict[str, set[str]] = {}
        for name, arm_data in data.get("arms", {}).items():
            bandit.arms[name] = ArmStats(
                pulls=arm_data["pulls"],
                total_reward=arm_data["total_reward"],
                mean_reward=arm_data["mean_reward"],
            )
            arm_techniques[name] = set(arm_data.get("techniques", []))
        return cls(
            bandit=bandit,
            arm_techniques=arm_techniques,
            _cold_started=True,
        )


# -----------------------------------------------------------------------
# FeedbackLoop
# -----------------------------------------------------------------------


@dataclass(slots=True)
class FeedbackLoop:
    """End-to-end RL feedback loop: select techniques, observe reward, learn.

    Persistence is via a JSON file so bandit state survives restarts.
    """

    selector: BanditSelector = field(default_factory=BanditSelector)
    reward_computer: RewardComputer = field(default_factory=RewardComputer)
    state_path: Path | None = None

    def __post_init__(self) -> None:
        if self.state_path is not None and self.state_path.exists():
            self._load()

    def before_query(
        self,
        query_type: str,
        corpus_size: int,
        registry: TechniqueRegistry | None = None,
    ) -> tuple[str, set[str]]:
        """Ask the bandit which techniques to enable.

        Returns ``(arm_name, technique_set)`` so the caller can pass
        the technique set as ``enabled_overrides`` to
        ``compose_pipeline``.
        """
        return self.selector.select_techniques(
            query_type, corpus_size, registry=registry,
        )

    def after_query(self, arm_name: str, answer: RAGAnswer) -> float:
        """Compute reward from *answer* and update the bandit.

        Returns the computed reward for observability.
        """
        reward = self.reward_computer.compute(answer)
        self.selector.record_reward(arm_name, reward)
        if self.state_path is not None:
            self._save()
        return reward

    # -- persistence -----------------------------------------------------

    def _save(self) -> None:
        """Persist bandit state to JSON."""
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.selector.to_dict()
        self.state_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def _load(self) -> None:
        """Load bandit state from JSON."""
        if self.state_path is None or not self.state_path.exists():
            return
        raw = self.state_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        self.selector = BanditSelector.from_dict(data)

    def save(self) -> None:
        """Public API — persist current state to disk."""
        self._save()

    def load(self) -> None:
        """Public API — reload state from disk."""
        self._load()
