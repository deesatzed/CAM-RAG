"""Adaptive retrieval parameter defaults with lightweight query/corpus tuning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol


class CorpusProfile(Protocol):
    """Minimal profile shape used for optional corpus-aware tuning."""

    complexity_level: str
    domain: str
    likely_question_types: list[str]


_TYPE_DEFAULTS = MappingProxyType({
    "specification": MappingProxyType({
        "dense_k": 10,
        "sparse_k": 20,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "final_k": 5,
        "diversity_lambda": 0.0,
        "min_confidence": 0.5,
        "novelty_threshold": 0.90,
        "max_chunks_per_doc": 3,
        "use_rrf_in_fractal": False,
    }),
    "summary": MappingProxyType({
        "dense_k": 20,
        "sparse_k": 30,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "final_k": 10,
        "diversity_lambda": 0.2,
        "min_confidence": 0.35,
        "novelty_threshold": 0.75,
        "max_chunks_per_doc": 5,
        "use_rrf_in_fractal": False,
    }),
    "logic": MappingProxyType({
        "dense_k": 15,
        "sparse_k": 25,
        "dense_weight": 0.65,
        "sparse_weight": 0.35,
        "final_k": 8,
        "diversity_lambda": 0.1,
        "min_confidence": 0.4,
        "novelty_threshold": 0.80,
        "max_chunks_per_doc": 4,
        "use_rrf_in_fractal": False,
    }),
    "synthesis": MappingProxyType({
        "dense_k": 30,
        "sparse_k": 50,
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "final_k": 15,
        "diversity_lambda": 0.4,
        "min_confidence": 0.3,
        "novelty_threshold": 0.65,
        "max_chunks_per_doc": 3,
        "use_rrf_in_fractal": True,
    }),
})


@dataclass(slots=True)
class AdaptiveParams:
    """Per-query retrieval parameters."""

    dense_k: int = 20
    sparse_k: int = 40
    rrf_k: float = 60.0
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    final_k: int = 10
    diversity_lambda: float = 0.3
    min_confidence: float = 0.4
    novelty_threshold: float = 0.80
    max_chunks_per_doc: int = 5
    use_rrf_in_fractal: bool = False


def compute_adaptive_params(
    query_type: str,
    query_text: str,
    corpus_profiles: list[CorpusProfile] | None = None,
    corpus_size: int = 0,
) -> AdaptiveParams:
    """Compute retrieval parameters tuned for query type and corpus hints."""

    defaults = _TYPE_DEFAULTS.get(query_type, _TYPE_DEFAULTS["summary"])
    params = AdaptiveParams(**defaults, rrf_k=60.0)

    if corpus_size > 50:
        params.dense_k = int(params.dense_k * 1.5)
        params.sparse_k = int(params.sparse_k * 1.5)
    if 0 < corpus_size < 10:
        params.final_k = min(params.final_k, corpus_size)
        params.min_confidence -= 0.1

    if corpus_profiles:
        _apply_profile_adjustments(params, corpus_profiles)

    if query_text.count("?") > 1:
        params.final_k += 5
        params.dense_k += 10
    if len(query_text) > 100:
        params.dense_k += 5

    weight_sum = params.dense_weight + params.sparse_weight
    if weight_sum > 0:
        params.dense_weight /= weight_sum
        params.sparse_weight /= weight_sum

    params.min_confidence = max(0.2, min(0.8, params.min_confidence))
    params.diversity_lambda = max(0.0, min(0.6, params.diversity_lambda))
    params.final_k = max(3, params.final_k)
    params.rrf_k = 60.0
    return params


def _apply_profile_adjustments(
    params: AdaptiveParams,
    corpus_profiles: list[CorpusProfile],
) -> None:
    complexity_scores = {"low": 0, "medium": 1, "high": 2}
    average_complexity = sum(
        complexity_scores.get(profile.complexity_level, 1) for profile in corpus_profiles
    ) / len(corpus_profiles)
    if average_complexity >= 1.5:
        params.dense_k = int(params.dense_k * 1.5)
        params.max_chunks_per_doc += 2

    domain_counts: dict[str, int] = {}
    for profile in corpus_profiles:
        domain = profile.domain.lower()
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if domain_counts and max(domain_counts, key=domain_counts.get) in {"medical", "clinical"}:
        params.min_confidence += 0.1

    specification_count = sum(
        1 for profile in corpus_profiles if "specification" in profile.likely_question_types
    )
    if specification_count > len(corpus_profiles) / 2:
        params.sparse_weight += 0.1
