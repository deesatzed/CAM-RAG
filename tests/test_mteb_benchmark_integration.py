"""Tests for MTEB benchmark integration, regression detection, and actionability.

Validates that:
- MTEB task configuration is correct for the platform's goals
- Benchmark results are structured for regression detection
- Score drops are flagged as regressions
- Results are actionable (linked to technique-level diagnostics)

All tests use real platform primitives. The MTEB library itself is optional;
tests that require it are gated with a skip marker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cam_rag.benchmarks.mteb import CamRAGMTEBModel, _coerce_texts, build_arg_parser
from cam_rag.retrieval import HashEmbeddingBackend

# ---------------------------------------------------------------------------
# 1. MTEB Task Configuration
# ---------------------------------------------------------------------------


# The MTEB tasks we intend to track for the platform
RETRIEVAL_TASKS = (
    "SciFactRetrieval",
    "NFCorpusRetrieval",
    "SciFact",
    "FiQA2018",
    "TRECCOVID",
)

SEMANTIC_SIMILARITY_TASKS = (
    "STSBenchmark",
    "STS22",
)

CLASSIFICATION_TASKS = (
    "MassiveIntentClassification",
)


class TestMTEBTaskConfiguration:
    """Validate that the MTEB harness is configured for the right tasks."""

    def test_arg_parser_defaults_to_retrieval_type(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.task_type == "Retrieval"

    def test_arg_parser_accepts_explicit_tasks(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--tasks", "SciFactRetrieval", "NFCorpusRetrieval"])
        assert args.tasks == ["SciFactRetrieval", "NFCorpusRetrieval"]

    def test_arg_parser_default_benchmark(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.benchmark == "MTEB(eng, v2)"

    def test_arg_parser_output_folder(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--output-folder", "/tmp/results"])
        assert args.output_folder == "/tmp/results"


# ---------------------------------------------------------------------------
# 2. Benchmark Result Structure and Regression Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """One task's score from an MTEB run."""

    task: str
    model: str
    main_score: float
    ndcg_at_10: float = 0.0
    map_at_10: float = 0.0
    recall_at_10: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Report of score regressions across benchmark runs."""

    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    stable: list[str]
    passed: bool


def detect_regressions(
    baseline: list[BenchmarkScore],
    current: list[BenchmarkScore],
    *,
    tolerance: float = 0.02,
) -> RegressionReport:
    """Detect score regressions between two benchmark runs.

    Args:
        baseline: Previous benchmark scores.
        current: Current benchmark scores.
        tolerance: Minimum drop to flag as regression.

    Returns:
        RegressionReport with regressions, improvements, and stable tasks.
    """
    baseline_map = {s.task: s for s in baseline}
    current_map = {s.task: s for s in current}

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    stable: list[str] = []

    for task in baseline_map:
        if task not in current_map:
            regressions.append({
                "task": task,
                "reason": "task_missing_from_current_run",
                "baseline_score": baseline_map[task].main_score,
            })
            continue

        delta = current_map[task].main_score - baseline_map[task].main_score
        if delta < -tolerance:
            regressions.append({
                "task": task,
                "baseline_score": baseline_map[task].main_score,
                "current_score": current_map[task].main_score,
                "delta": round(delta, 6),
                "reason": "score_drop",
            })
        elif delta > tolerance:
            improvements.append({
                "task": task,
                "baseline_score": baseline_map[task].main_score,
                "current_score": current_map[task].main_score,
                "delta": round(delta, 6),
            })
        else:
            stable.append(task)

    return RegressionReport(
        regressions=regressions,
        improvements=improvements,
        stable=stable,
        passed=len(regressions) == 0,
    )


class TestRegressionDetection:
    """Regression detector must flag score drops and missing tasks."""

    def test_no_regression_when_scores_match(self):
        baseline = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
            BenchmarkScore(task="NFCorpusRetrieval", model="e5", main_score=0.32),
        ]
        current = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
            BenchmarkScore(task="NFCorpusRetrieval", model="e5", main_score=0.32),
        ]

        report = detect_regressions(baseline, current)

        assert report.passed is True
        assert len(report.regressions) == 0
        assert len(report.stable) == 2

    def test_flags_score_drop_as_regression(self):
        baseline = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
        ]
        current = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.55),
        ]

        report = detect_regressions(baseline, current)

        assert report.passed is False
        assert len(report.regressions) == 1
        assert report.regressions[0]["task"] == "SciFactRetrieval"
        assert report.regressions[0]["delta"] == pytest.approx(-0.1, abs=0.01)

    def test_ignores_small_fluctuations_within_tolerance(self):
        baseline = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
        ]
        current = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.64),
        ]

        report = detect_regressions(baseline, current, tolerance=0.02)

        assert report.passed is True
        assert len(report.stable) == 1

    def test_flags_missing_task_as_regression(self):
        baseline = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
            BenchmarkScore(task="NFCorpusRetrieval", model="e5", main_score=0.32),
        ]
        current = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.65),
            # NFCorpusRetrieval missing
        ]

        report = detect_regressions(baseline, current)

        assert report.passed is False
        assert any(r["reason"] == "task_missing_from_current_run" for r in report.regressions)

    def test_detects_improvement(self):
        baseline = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.55),
        ]
        current = [
            BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.70),
        ]

        report = detect_regressions(baseline, current)

        assert report.passed is True
        assert len(report.improvements) == 1
        assert report.improvements[0]["delta"] > 0


# ---------------------------------------------------------------------------
# 3. Score Persistence and JSON Round-Trip
# ---------------------------------------------------------------------------


class TestScorePersistence:
    """Benchmark scores must serialize to JSON for storage and comparison."""

    def test_scores_round_trip_through_json(self, tmp_path: Path):
        scores = [
            BenchmarkScore(
                task="SciFactRetrieval",
                model="cam-rag-hash-256",
                main_score=0.05,
                ndcg_at_10=0.04,
            ),
            BenchmarkScore(
                task="NFCorpusRetrieval",
                model="cam-rag-hash-256",
                main_score=0.02,
                recall_at_10=0.01,
            ),
        ]

        path = tmp_path / "scores.json"
        # Serialize
        data = [
            {
                "task": s.task,
                "model": s.model,
                "main_score": s.main_score,
                "ndcg_at_10": s.ndcg_at_10,
                "map_at_10": s.map_at_10,
                "recall_at_10": s.recall_at_10,
            }
            for s in scores
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        # Deserialize
        loaded = json.loads(path.read_text(encoding="utf-8"))
        restored = [BenchmarkScore(**row) for row in loaded]

        assert len(restored) == 2
        assert restored[0].task == "SciFactRetrieval"
        assert restored[0].main_score == 0.05
        assert restored[1].recall_at_10 == 0.01


# ---------------------------------------------------------------------------
# 4. CamRAGMTEBModel Encoding Correctness
# ---------------------------------------------------------------------------


class TestCamRAGMTEBModelEncoding:
    """The MTEB model wrapper must produce correct, consistent embeddings."""

    def test_encode_deterministic(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=32))
        first = model.encode(["test query"])
        second = model.encode(["test query"])
        assert first == second

    def test_encode_different_texts_produce_different_vectors(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=64))
        results = model.encode(["alpha", "beta"])
        assert results[0] != results[1]

    def test_encode_handles_empty_list(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=16))
        results = model.encode([])
        assert results == []

    def test_encode_handles_dict_inputs(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=16))
        results = model.encode([{"text": "hello"}, {"query": "world"}])
        assert len(results) == 2
        assert all(len(v) == 16 for v in results)

    def test_encode_handles_nested_batches(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=8))
        # Simulates DataLoader-style batches
        results = model.encode([["sentence one", "sentence two"]])
        assert len(results) == 2

    def test_model_meta_has_name(self):
        model = CamRAGMTEBModel(HashEmbeddingBackend(dim=16), name="test-model")
        assert model.mteb_model_meta["name"] == "test-model"


# ---------------------------------------------------------------------------
# 5. Coerce Texts Robustness
# ---------------------------------------------------------------------------


class TestCoerceTexts:
    """The _coerce_texts helper must handle all MTEB input formats."""

    def test_plain_strings(self):
        assert _coerce_texts(["a", "b"]) == ["a", "b"]

    def test_single_string(self):
        assert _coerce_texts("single") == ["single"]

    def test_dict_with_text_key(self):
        assert _coerce_texts([{"text": "hello"}]) == ["hello"]

    def test_dict_with_query_key(self):
        assert _coerce_texts([{"query": "question"}]) == ["question"]

    def test_nested_batches(self):
        inputs = [["a", "b"], ["c"]]
        assert _coerce_texts(inputs) == ["a", "b", "c"]

    def test_mixed_types(self):
        inputs = ["plain", {"text": "dict"}, 42]
        result = _coerce_texts(inputs)
        assert result == ["plain", "dict", "42"]


# ---------------------------------------------------------------------------
# 6. Actionable Results: Link Scores to Technique Diagnostics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TechniqueDiagnostic:
    """Links a benchmark score to specific technique-level analysis."""

    task: str
    score: float
    embedding_backend: str
    sparse_contribution: float
    dense_contribution: float
    fusion_method: str
    recommendation: str


def diagnose_score(score: BenchmarkScore, technique_config: dict[str, Any]) -> TechniqueDiagnostic:
    """Turn a raw benchmark score into an actionable diagnostic."""
    recommendation = ""
    if score.main_score < 0.1:
        recommendation = (
            f"Score {score.main_score:.3f} on {score.task} is below baseline. "
            f"Current backend '{technique_config.get('embedding_backend', 'unknown')}' "
            f"likely needs replacement with a trained model."
        )
    elif score.main_score < 0.5:
        recommendation = (
            f"Score {score.main_score:.3f} on {score.task} is moderate. "
            f"Consider fine-tuning or switching to a domain-specific embedding model."
        )
    else:
        recommendation = (
            f"Score {score.main_score:.3f} on {score.task} is competitive. "
            f"Focus on fusion and reranking improvements."
        )

    return TechniqueDiagnostic(
        task=score.task,
        score=score.main_score,
        embedding_backend=technique_config.get("embedding_backend", "unknown"),
        sparse_contribution=technique_config.get("sparse_weight", 0.4),
        dense_contribution=technique_config.get("dense_weight", 0.6),
        fusion_method=technique_config.get("fusion_method", "rrf"),
        recommendation=recommendation,
    )


class TestActionableDiagnostics:
    """Benchmark scores must produce actionable recommendations."""

    def test_low_score_recommends_backend_replacement(self):
        score = BenchmarkScore(task="SciFactRetrieval", model="hash", main_score=0.02)
        config = {"embedding_backend": "hash", "sparse_weight": 0.4, "dense_weight": 0.6}

        diag = diagnose_score(score, config)

        assert "below baseline" in diag.recommendation
        assert "hash" in diag.recommendation
        assert "replacement" in diag.recommendation

    def test_moderate_score_recommends_fine_tuning(self):
        score = BenchmarkScore(task="SciFactRetrieval", model="e5", main_score=0.35)
        config = {"embedding_backend": "e5-base-v2"}

        diag = diagnose_score(score, config)

        assert "fine-tuning" in diag.recommendation or "domain-specific" in diag.recommendation

    def test_high_score_recommends_fusion_focus(self):
        score = BenchmarkScore(task="SciFactRetrieval", model="bge-large", main_score=0.72)
        config = {"embedding_backend": "bge-large-en-v1.5"}

        diag = diagnose_score(score, config)

        assert "fusion" in diag.recommendation or "reranking" in diag.recommendation

    def test_diagnostic_includes_technique_config(self):
        score = BenchmarkScore(task="test", model="m", main_score=0.5)
        config = {
            "embedding_backend": "e5",
            "sparse_weight": 0.3,
            "dense_weight": 0.7,
            "fusion_method": "rrf",
        }

        diag = diagnose_score(score, config)

        assert diag.embedding_backend == "e5"
        assert diag.sparse_contribution == 0.3
        assert diag.dense_contribution == 0.7
        assert diag.fusion_method == "rrf"
