"""Regression detection for MTEB benchmark scores.

Compares current benchmark scores against stored baselines and flags
regressions that exceed a configurable threshold.  This prevents
pipeline changes from silently degrading retrieval quality.

Usage::

    guardian = RegressionGuardian("benchmarks/mteb/baselines/")
    report = guardian.check({
        "SciFact:qwen3-8b-pipeline": 0.745,
    })
    if not report.passed:
        for reg in report.regressions:
            print(f"REGRESSION: {reg['task']} dropped {reg['delta']:.4f}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASELINES_FILENAME = "regression_baselines.json"


@dataclass(frozen=True, slots=True)
class BenchmarkBaseline:
    """A single known-good benchmark score."""

    task: str
    config: str
    metric: str
    value: float
    timestamp: str = ""


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Result of comparing current scores against baselines."""

    passed: bool
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    unchanged: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the report."""
        parts = []
        if self.regressions:
            parts.append(
                f"{len(self.regressions)} REGRESSION(S): "
                + ", ".join(
                    f"{r['key']} ({r['delta']:+.5f})" for r in self.regressions
                )
            )
        if self.improvements:
            parts.append(
                f"{len(self.improvements)} improvement(s): "
                + ", ".join(
                    f"{r['key']} ({r['delta']:+.5f})" for r in self.improvements
                )
            )
        if self.unchanged:
            parts.append(f"{len(self.unchanged)} unchanged")
        return "; ".join(parts) or "No baselines to compare"


class RegressionGuardian:
    """Compare benchmark scores against baselines and detect regressions.

    Parameters
    ----------
    baselines_path:
        Directory containing ``regression_baselines.json``.
    threshold:
        Minimum score drop (absolute) to flag as a regression.
        Default 0.005 (0.5% of nDCG@10 scale).
    """

    def __init__(
        self,
        baselines_path: str = "benchmarks/mteb/baselines/",
        threshold: float = 0.005,
    ) -> None:
        self._baselines_dir = Path(baselines_path)
        self._threshold = threshold
        self._baselines = self._load_baselines()

    def _load_baselines(self) -> dict[str, BenchmarkBaseline]:
        """Load baselines from JSON file."""
        path = self._baselines_dir / _BASELINES_FILENAME
        if not path.exists():
            logger.warning("No baselines file found at %s", path)
            return {}

        data = json.loads(path.read_text(encoding="utf-8"))
        baselines: dict[str, BenchmarkBaseline] = {}
        for entry in data.get("baselines", []):
            key = f"{entry['task']}:{entry['config']}"
            baselines[key] = BenchmarkBaseline(
                task=entry["task"],
                config=entry["config"],
                metric=entry.get("metric", "ndcg_10"),
                value=entry["value"],
                timestamp=entry.get("timestamp", ""),
            )
        return baselines

    def check(
        self, current_scores: dict[str, float]
    ) -> RegressionReport:
        """Compare current scores against baselines.

        Parameters
        ----------
        current_scores:
            Dict mapping ``"task:config"`` keys to current metric values.

        Returns
        -------
        RegressionReport
            Report with regressions, improvements, and unchanged entries.
        """
        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []
        unchanged: list[dict[str, Any]] = []

        for key, current_value in current_scores.items():
            baseline = self._baselines.get(key)
            if baseline is None:
                logger.debug("No baseline for %s, skipping", key)
                continue

            delta = current_value - baseline.value
            entry = {
                "key": key,
                "task": baseline.task,
                "config": baseline.config,
                "baseline": baseline.value,
                "current": current_value,
                "delta": delta,
            }

            if delta < -self._threshold:
                regressions.append(entry)
            elif delta > self._threshold:
                improvements.append(entry)
            else:
                unchanged.append(entry)

        passed = len(regressions) == 0
        return RegressionReport(
            passed=passed,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
        )

    def update_baselines(
        self, scores: dict[str, float], config: str
    ) -> None:
        """Update baselines with new scores when they improve.

        Only updates entries where the new score exceeds the existing
        baseline.  Creates new entries for previously unseen tasks.

        Parameters
        ----------
        scores:
            Dict mapping task names to metric values.
        config:
            Configuration label for these scores.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        updated = False

        for task, value in scores.items():
            key = f"{task}:{config}"
            existing = self._baselines.get(key)
            if existing is None or value > existing.value:
                self._baselines[key] = BenchmarkBaseline(
                    task=task,
                    config=config,
                    metric="ndcg_10",
                    value=value,
                    timestamp=timestamp,
                )
                updated = True

        if updated:
            self._save_baselines()

    def _save_baselines(self) -> None:
        """Write baselines to JSON file."""
        self._baselines_dir.mkdir(parents=True, exist_ok=True)
        path = self._baselines_dir / _BASELINES_FILENAME
        data = {
            "baselines": [
                {
                    "task": b.task,
                    "config": b.config,
                    "metric": b.metric,
                    "value": b.value,
                    "timestamp": b.timestamp,
                }
                for b in sorted(
                    self._baselines.values(),
                    key=lambda b: (b.task, b.config),
                )
            ]
        }
        path.write_text(
            json.dumps(data, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @property
    def baseline_count(self) -> int:
        """Number of loaded baselines."""
        return len(self._baselines)
