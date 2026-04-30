"""Tests for RegressionGuardian benchmark regression detection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cam_rag.benchmarks.regression import (
    BenchmarkBaseline,
    RegressionGuardian,
    RegressionReport,
)


def _write_baselines(tmpdir: Path, baselines: list[dict]) -> str:
    """Write a regression_baselines.json to a temp dir."""
    path = tmpdir / "regression_baselines.json"
    path.write_text(
        json.dumps({"baselines": baselines}, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(tmpdir)


class TestBenchmarkBaseline:
    def test_fields(self) -> None:
        b = BenchmarkBaseline(
            task="SciFact",
            config="qwen3-8b-embed-only",
            metric="ndcg_10",
            value=0.768,
            timestamp="2026-04-30T00:00:00+00:00",
        )
        assert b.task == "SciFact"
        assert b.value == 0.768


class TestRegressionReport:
    def test_summary_with_regressions(self) -> None:
        report = RegressionReport(
            passed=False,
            regressions=[
                {"key": "SciFact:test", "delta": -0.02},
            ],
        )
        assert "REGRESSION" in report.summary()

    def test_summary_with_improvements(self) -> None:
        report = RegressionReport(
            passed=True,
            improvements=[
                {"key": "SciFact:test", "delta": 0.01},
            ],
        )
        assert "improvement" in report.summary()

    def test_summary_empty(self) -> None:
        report = RegressionReport(passed=True)
        assert report.summary() == "No baselines to compare"


class TestRegressionGuardian:
    def test_detects_regression_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            report = guardian.check({"SciFact:test": 0.750})
            assert not report.passed
            assert len(report.regressions) == 1
            assert report.regressions[0]["delta"] < -0.005

    def test_passes_when_score_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            report = guardian.check({"SciFact:test": 0.768})
            assert report.passed
            assert len(report.unchanged) == 1

    def test_passes_when_score_improves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            report = guardian.check({"SciFact:test": 0.790})
            assert report.passed
            assert len(report.improvements) == 1

    def test_small_regression_below_threshold_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            # 0.764 is -0.004, below 0.005 threshold
            report = guardian.check({"SciFact:test": 0.764})
            assert report.passed

    def test_handles_missing_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # No baselines file at all
            guardian = RegressionGuardian(tmpdir, threshold=0.005)
            assert guardian.baseline_count == 0
            report = guardian.check({"SciFact:test": 0.768})
            assert report.passed
            assert len(report.regressions) == 0

    def test_update_baselines_on_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            # Update with improvement
            guardian.update_baselines({"SciFact": 0.790}, config="test")
            # Reload and verify
            guardian2 = RegressionGuardian(baselines_dir, threshold=0.005)
            report = guardian2.check({"SciFact:test": 0.785})
            # 0.785 < 0.790 baseline → regression
            assert not report.passed

    def test_update_does_not_regress_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            # Try to update with worse score — should NOT overwrite
            guardian.update_baselines({"SciFact": 0.750}, config="test")
            guardian2 = RegressionGuardian(baselines_dir, threshold=0.005)
            # Baseline should still be 0.768
            report = guardian2.check({"SciFact:test": 0.768})
            assert report.passed

    def test_loads_real_baselines_file(self) -> None:
        """Load the actual regression_baselines.json from the project."""
        guardian = RegressionGuardian(
            "benchmarks/mteb/baselines/", threshold=0.005
        )
        assert guardian.baseline_count > 0

    def test_multiple_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            baselines_dir = _write_baselines(tmppath, [
                {"task": "SciFact", "config": "test", "metric": "ndcg_10", "value": 0.768},
                {"task": "NFCorpus", "config": "test", "metric": "ndcg_10", "value": 0.384},
            ])
            guardian = RegressionGuardian(baselines_dir, threshold=0.005)
            report = guardian.check({
                "SciFact:test": 0.780,   # improvement
                "NFCorpus:test": 0.370,  # regression
            })
            assert not report.passed
            assert len(report.improvements) == 1
            assert len(report.regressions) == 1
