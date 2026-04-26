from dataclasses import dataclass

import pytest

from cam_rag.retrieval import compute_adaptive_params


@dataclass
class Profile:
    complexity_level: str
    domain: str
    likely_question_types: list[str]


def test_adaptive_params_use_type_defaults_and_normalized_weights():
    params = compute_adaptive_params("specification", "What is the exact setting?")

    assert params.dense_k == 10
    assert params.sparse_k == 20
    assert params.final_k == 5
    assert params.dense_weight + params.sparse_weight == 1.0
    assert params.rrf_k == 60.0


def test_adaptive_params_expand_for_large_complex_corpus_and_multi_question_query():
    profiles = [
        Profile("high", "medical", ["specification"]),
        Profile("high", "clinical", ["specification"]),
        Profile("medium", "medical", ["summary"]),
    ]

    params = compute_adaptive_params(
        "summary",
        "What changed? Why does it matter?",
        corpus_profiles=profiles,
        corpus_size=100,
    )

    assert params.dense_k == 55
    assert params.sparse_k == 45
    assert params.final_k == 15
    assert params.max_chunks_per_doc == 7
    assert params.min_confidence == pytest.approx(0.45)
    assert params.sparse_weight > 0.4


def test_adaptive_params_clamp_small_corpus_final_k():
    params = compute_adaptive_params("synthesis", "summarize", corpus_size=2)

    assert params.final_k == 3
    assert params.min_confidence == 0.2
