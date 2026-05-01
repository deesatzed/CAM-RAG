"""Tests for PipelineCalibrator and CalibrationResult."""

from __future__ import annotations

import pytest

from cam_rag.pipeline.calibration import CalibrationResult, PipelineCalibrator
from cam_rag.retrieval.dense import HashEmbeddingBackend
from cam_rag.retrieval.models import RetrievalDocument
from cam_rag.retrieval.sparse import tokenize


def _make_docs(n: int, topic_words: int = 5) -> list[RetrievalDocument]:
    """Create n documents with varied content for calibration testing."""
    topics = [
        "machine learning",
        "data science",
        "neural networks",
        "deep learning",
        "natural language",
        "computer vision",
        "reinforcement learning",
        "gradient descent",
        "backpropagation",
        "transformer models",
    ]
    docs: list[RetrievalDocument] = []
    for i in range(n):
        topic = topics[i % len(topics)]
        words = topic.split() + [f"word{j}" for j in range(topic_words)]
        text = " ".join(words * 3)  # repeat to give enough tokens
        docs.append(RetrievalDocument(doc_id=f"doc_{i}", text=text))
    return docs


# ------------------------------------------------------------------ #
# TestCalibrationResult
# ------------------------------------------------------------------ #


class TestCalibrationResult:
    def test_frozen(self) -> None:
        result = CalibrationResult(
            dense_weight=0.6,
            sparse_weight=0.4,
            retrieval_depth=100,
            rrf_k=60.0,
            calibration_score=0.5,
            grid_points_evaluated=54,
            holdout_size=10,
        )
        with pytest.raises(AttributeError):
            result.dense_weight = 0.9  # type: ignore[misc]

    def test_slots(self) -> None:
        assert hasattr(CalibrationResult, "__slots__")

    def test_fields(self) -> None:
        result = CalibrationResult(
            dense_weight=0.7,
            sparse_weight=0.3,
            retrieval_depth=150,
            rrf_k=90.0,
            calibration_score=0.8,
            grid_points_evaluated=54,
            holdout_size=16,
        )
        assert result.dense_weight == 0.7
        assert result.sparse_weight == 0.3
        assert result.retrieval_depth == 150
        assert result.rrf_k == 90.0
        assert result.calibration_score == 0.8
        assert result.grid_points_evaluated == 54
        assert result.holdout_size == 16


# ------------------------------------------------------------------ #
# TestPseudoQueryGeneration
# ------------------------------------------------------------------ #


class TestPseudoQueryGeneration:
    def test_generates_queries(self) -> None:
        docs = _make_docs(100)
        calibrator = PipelineCalibrator(seed=42, max_pseudo_queries=20)
        backend = HashEmbeddingBackend(dim=64)
        result = calibrator.calibrate(docs, backend, tokenize)
        # If queries were generated, holdout_size > 0
        assert result.holdout_size > 0

    def test_deterministic_with_seed(self) -> None:
        docs = _make_docs(100)
        backend = HashEmbeddingBackend(dim=64)
        cal_a = PipelineCalibrator(seed=99, max_pseudo_queries=20)
        cal_b = PipelineCalibrator(seed=99, max_pseudo_queries=20)
        result_a = cal_a.calibrate(docs, backend, tokenize, seed=99)
        result_b = cal_b.calibrate(docs, backend, tokenize, seed=99)
        assert result_a.dense_weight == result_b.dense_weight
        assert result_a.sparse_weight == result_b.sparse_weight
        assert result_a.retrieval_depth == result_b.retrieval_depth
        assert result_a.rrf_k == result_b.rrf_k
        assert result_a.calibration_score == result_b.calibration_score

    def test_queries_contain_doc_terms(self) -> None:
        """Pseudo-query terms should come from the holdout document."""
        docs = _make_docs(100)
        calibrator = PipelineCalibrator(seed=42, max_pseudo_queries=20)

        # Access internal method to inspect pseudo-queries directly
        import random as _random

        rng = _random.Random(42)
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        train_docs = [docs[i] for i in indices[:split]]
        holdout_docs = [docs[i] for i in indices[split:]]

        rng2 = _random.Random(42)
        queries = calibrator._generate_pseudo_queries(
            train_docs, holdout_docs, tokenize, rng2
        )
        assert len(queries) > 0
        for query_text, expected_doc_id in queries:
            # Find the holdout doc
            holdout_doc = next(d for d in holdout_docs if d.doc_id == expected_doc_id)
            holdout_tokens = set(tokenize(holdout_doc.text))
            query_tokens = set(tokenize(query_text))
            # Every query token should come from the holdout doc
            assert query_tokens.issubset(holdout_tokens), (
                f"Query tokens {query_tokens - holdout_tokens} "
                f"not in holdout doc {expected_doc_id}"
            )


# ------------------------------------------------------------------ #
# TestPipelineCalibrator
# ------------------------------------------------------------------ #


class TestPipelineCalibrator:
    def test_calibrate_returns_result(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert isinstance(result, CalibrationResult)
        assert result.dense_weight > 0
        assert result.retrieval_depth > 0
        assert result.rrf_k > 0

    def test_weights_valid(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert abs(result.dense_weight + result.sparse_weight - 1.0) < 0.01

    def test_depth_in_valid_range(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert result.retrieval_depth in {50, 100, 150}

    def test_rrf_k_positive(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert result.rrf_k > 0

    def test_calibration_score_between_0_and_1(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert 0.0 <= result.calibration_score <= 1.0

    def test_caches_result(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result_a = calibrator.calibrate(docs, backend, tokenize)
        result_b = calibrator.calibrate(docs, backend, tokenize)
        # Same object returned from cache
        assert result_a is result_b

    def test_too_small_corpus_returns_defaults(self) -> None:
        docs = _make_docs(30)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        assert result.dense_weight == 0.6
        assert result.sparse_weight == 0.4
        assert result.retrieval_depth == 100
        assert result.rrf_k == 60.0
        assert result.calibration_score == 0.0
        assert result.grid_points_evaluated == 0

    def test_deterministic(self) -> None:
        docs = _make_docs(100)
        backend = HashEmbeddingBackend(dim=64)
        cal_a = PipelineCalibrator(seed=7)
        cal_b = PipelineCalibrator(seed=7)
        result_a = cal_a.calibrate(docs, backend, tokenize, seed=7)
        result_b = cal_b.calibrate(docs, backend, tokenize, seed=7)
        assert result_a == result_b

    def test_grid_points_evaluated(self) -> None:
        docs = _make_docs(80)
        backend = HashEmbeddingBackend(dim=64)
        calibrator = PipelineCalibrator(seed=42)
        result = calibrator.calibrate(docs, backend, tokenize)
        # 6 dense_weights * 3 depths * 3 rrf_ks = 54
        assert result.grid_points_evaluated == 54
