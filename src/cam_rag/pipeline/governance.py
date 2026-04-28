"""Technique fitness scoring and lifecycle governance.

Adapted from CAM-Pulse's fitness.py and lifecycle.py patterns for the
cam-rag-platform.  Each retrieval technique is scored across six weighted
dimensions; an exponential moving average (EMA) tracks per-technique
fitness over time.  The lifecycle state machine promotes high-performing
techniques and gracefully retires underperformers.

Dimensions and weights (sum = 1.0):
    1. retrieval_quality  (0.30) -- quality of retrieved evidence
    2. query_type_affinity (0.20) -- match to query type
    3. confidence_lift     (0.15) -- improvement in answer confidence
    4. diversity_contribution (0.15) -- diversity added to the result set
    5. latency_efficiency  (0.10) -- speed relative to budget
    6. recency             (0.10) -- freshness of the technique's track record
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimension weights
# ---------------------------------------------------------------------------
W_RETRIEVAL_QUALITY = 0.30
W_QUERY_TYPE_AFFINITY = 0.20
W_CONFIDENCE_LIFT = 0.15
W_DIVERSITY_CONTRIBUTION = 0.15
W_LATENCY_EFFICIENCY = 0.10
W_RECENCY = 0.10

# ---------------------------------------------------------------------------
# EMA defaults
# ---------------------------------------------------------------------------
DEFAULT_EMA_ALPHA = 0.3

# ---------------------------------------------------------------------------
# Lifecycle thresholds
# ---------------------------------------------------------------------------
EMBRYONIC_OBSERVATION_MIN = 10
ACTIVE_TO_PREFERRED_FITNESS = 0.7
PREFERRED_TO_DECLINING_FITNESS = 0.4
DECLINING_TO_DISABLED_FITNESS = 0.2
DECLINING_TO_DISABLED_CONSECUTIVE = 5


def _clamp01(value: float) -> float:
    """Clamp *value* to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ===================================================================
# FitnessScore
# ===================================================================


@dataclass(frozen=True, slots=True)
class FitnessScore:
    """Six-dimensional fitness score for a retrieval technique.

    Every dimension is a float in [0.0, 1.0].  The ``overall`` property
    computes the weighted sum using the module-level weights.
    """

    retrieval_quality: float
    query_type_affinity: float
    confidence_lift: float
    diversity_contribution: float
    latency_efficiency: float
    recency: float

    def __post_init__(self) -> None:
        for name in (
            "retrieval_quality",
            "query_type_affinity",
            "confidence_lift",
            "diversity_contribution",
            "latency_efficiency",
            "recency",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                raise TypeError(
                    f"{name} must be a float, got {type(val).__name__}"
                )
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"{name} must be in [0.0, 1.0], got {val}"
                )

    @property
    def overall(self) -> float:
        """Weighted sum of all six dimensions."""
        return (
            W_RETRIEVAL_QUALITY * self.retrieval_quality
            + W_QUERY_TYPE_AFFINITY * self.query_type_affinity
            + W_CONFIDENCE_LIFT * self.confidence_lift
            + W_DIVERSITY_CONTRIBUTION * self.diversity_contribution
            + W_LATENCY_EFFICIENCY * self.latency_efficiency
            + W_RECENCY * self.recency
        )


# ===================================================================
# FitnessTracker
# ===================================================================


@dataclass(slots=True)
class FitnessTracker:
    """Per-technique EMA fitness tracker.

    Uses an exponential moving average so recent observations carry more
    weight.  The ``alpha`` parameter controls how fast the EMA adapts:
    alpha=0.3 means 30% weight on the newest observation.
    """

    alpha: float = DEFAULT_EMA_ALPHA
    _ema: dict[str, float] = field(default_factory=dict)
    _observation_counts: dict[str, int] = field(default_factory=dict)

    def record(self, technique_name: str, score: FitnessScore) -> None:
        """Record an observation for *technique_name*.

        Updates the EMA fitness value.  On the first observation the EMA
        is seeded with the overall score directly.
        """
        overall = score.overall
        prev = self._ema.get(technique_name)
        if prev is None:
            self._ema[technique_name] = overall
        else:
            self._ema[technique_name] = (
                self.alpha * overall + (1.0 - self.alpha) * prev
            )
        self._observation_counts[technique_name] = (
            self._observation_counts.get(technique_name, 0) + 1
        )

    def get_fitness(self, technique_name: str) -> float:
        """Return the current EMA fitness for *technique_name*.

        Returns 0.0 if the technique has never been observed.
        """
        return self._ema.get(technique_name, 0.0)

    def observation_count(self, technique_name: str) -> int:
        """Return how many observations have been recorded."""
        return self._observation_counts.get(technique_name, 0)

    def rank_techniques(self) -> list[tuple[str, float]]:
        """Return all tracked techniques sorted by fitness descending."""
        return sorted(
            self._ema.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )

    @property
    def technique_names(self) -> set[str]:
        """Set of all technique names that have at least one observation."""
        return set(self._ema)


# ===================================================================
# TechniqueState & LifecycleManager
# ===================================================================


class TechniqueState(enum.Enum):
    """Lifecycle states for a retrieval technique."""

    EMBRYONIC = "embryonic"
    ACTIVE = "active"
    PREFERRED = "preferred"
    DECLINING = "declining"
    DISABLED = "disabled"


@dataclass(slots=True)
class LifecycleManager:
    """Manages lifecycle state transitions for retrieval techniques.

    State machine:
        embryonic  -> active     : after ``embryonic_min`` observations
        active     -> preferred  : fitness > ``preferred_threshold``
        preferred  -> declining  : fitness < ``declining_threshold``
        declining  -> disabled   : fitness < ``disabled_threshold``
                                   for ``disabled_consecutive`` checks
        disabled   -> embryonic  : manual ``reactivate()`` call
    """

    embryonic_min: int = EMBRYONIC_OBSERVATION_MIN
    preferred_threshold: float = ACTIVE_TO_PREFERRED_FITNESS
    declining_threshold: float = PREFERRED_TO_DECLINING_FITNESS
    disabled_threshold: float = DECLINING_TO_DISABLED_FITNESS
    disabled_consecutive: int = DECLINING_TO_DISABLED_CONSECUTIVE

    _states: dict[str, TechniqueState] = field(default_factory=dict)
    _declining_streak: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, technique_name: str) -> TechniqueState:
        """Return the lifecycle state for *technique_name*.

        Defaults to ``EMBRYONIC`` for unknown techniques.
        """
        return self._states.get(technique_name, TechniqueState.EMBRYONIC)

    def evaluate_transitions(
        self, tracker: FitnessTracker
    ) -> dict[str, tuple[TechniqueState, TechniqueState]]:
        """Check all tracked techniques and apply state transitions.

        Returns a dict mapping technique names to ``(old_state, new_state)``
        tuples for every technique that transitioned.
        """
        transitions: dict[str, tuple[TechniqueState, TechniqueState]] = {}
        for name in tracker.technique_names:
            old = self.get_state(name)
            new = self._evaluate_one(name, tracker)
            if new is not None and new != old:
                self._states[name] = new
                transitions[name] = (old, new)
                logger.info(
                    "Technique %s transitioned %s -> %s",
                    name,
                    old.value,
                    new.value,
                )
        return transitions

    def reactivate(self, technique_name: str) -> None:
        """Manually re-enable a disabled technique.

        Resets it to ``EMBRYONIC`` so it must re-prove itself.
        """
        old = self.get_state(technique_name)
        self._states[technique_name] = TechniqueState.EMBRYONIC
        self._declining_streak.pop(technique_name, None)
        logger.info(
            "Technique %s reactivated: %s -> embryonic",
            technique_name,
            old.value,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_one(
        self, name: str, tracker: FitnessTracker
    ) -> TechniqueState | None:
        """Evaluate a single technique and return its new state or None."""
        current = self.get_state(name)
        fitness = tracker.get_fitness(name)
        obs = tracker.observation_count(name)

        if current == TechniqueState.DISABLED:
            # Disabled is terminal until manual reactivation.
            return None

        if current == TechniqueState.EMBRYONIC:
            if obs >= self.embryonic_min:
                return TechniqueState.ACTIVE
            return None

        if current == TechniqueState.ACTIVE:
            if fitness > self.preferred_threshold:
                return TechniqueState.PREFERRED
            return None

        if current == TechniqueState.PREFERRED:
            if fitness < self.declining_threshold:
                return TechniqueState.DECLINING
            return None

        if current == TechniqueState.DECLINING:
            if fitness < self.disabled_threshold:
                streak = self._declining_streak.get(name, 0) + 1
                self._declining_streak[name] = streak
                if streak >= self.disabled_consecutive:
                    return TechniqueState.DISABLED
            else:
                # Reset streak when fitness recovers above the threshold.
                self._declining_streak[name] = 0
            return None

        return None  # pragma: no cover
