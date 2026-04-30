"""Tests for LocalCrossEncoderBackend.

Uses a real cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)
for testing.  No mocks.
"""

from __future__ import annotations

import math

import pytest

from cam_rag.reranking.cross_encoder import LocalCrossEncoderBackend

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.fixture(scope="module")
def backend():
    """Load the cross-encoder once for all tests in this module."""
    return LocalCrossEncoderBackend(MODEL_NAME, device="cpu")


class TestLocalCrossEncoderBackend:
    def test_loads_model(self, backend):
        assert backend._model is not None

    def test_rerank_returns_correct_count(self, backend):
        passages = [
            "Aspirin is an anti-inflammatory drug.",
            "Python is a programming language.",
            "Ibuprofen reduces pain and inflammation.",
        ]
        scores = backend.rerank("What reduces inflammation?", passages)
        assert len(scores) == 3

    def test_scores_are_finite(self, backend):
        passages = [
            "The quick brown fox jumps over the lazy dog.",
            "Medical research advances in immunology.",
        ]
        scores = backend.rerank("Tell me about immunology", passages)
        for s in scores:
            assert math.isfinite(s), f"Non-finite score: {s}"

    def test_relevant_passage_scores_higher(self, backend):
        passages = [
            "Aspirin inhibits cyclooxygenase enzymes to reduce inflammation.",
            "The capital of France is Paris.",
        ]
        scores = backend.rerank("How does aspirin reduce inflammation?", passages)
        # The relevant passage should score higher
        assert scores[0] > scores[1], (
            f"Expected relevant passage to score higher: {scores}"
        )

    def test_empty_passages_returns_empty(self, backend):
        scores = backend.rerank("some query", [])
        assert scores == []

    def test_single_passage(self, backend):
        scores = backend.rerank("query", ["single passage"])
        assert len(scores) == 1
        assert math.isfinite(scores[0])
