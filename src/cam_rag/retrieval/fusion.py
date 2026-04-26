"""Reciprocal Rank Fusion for combining ranked retrieval lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cam_rag.retrieval.models import FusedResult, RetrievalResult


def rrf_fuse(
    *ranked_lists: Sequence[RetrievalResult],
    rrf_k: float = 60.0,
    weights: Sequence[float] | Mapping[str, float] | None = None,
    names: Sequence[str] | None = None,
) -> list[FusedResult]:
    """Fuse one or more ranked result lists with Reciprocal Rank Fusion."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if names is None:
        names = [f"source_{index + 1}" for index in range(len(ranked_lists))]
    if len(names) != len(ranked_lists):
        raise ValueError("names must match the number of ranked lists")

    resolved_weights = _resolve_weights(names, weights)
    fused: dict[str, dict[str, object]] = {}

    for source_name, results in zip(names, ranked_lists, strict=True):
        weight = resolved_weights[source_name]
        seen_in_source: set[str] = set()
        for fallback_rank, result in enumerate(results, start=1):
            if result.doc_id in seen_in_source:
                continue
            seen_in_source.add(result.doc_id)
            rank = result.rank if result.rank > 0 else fallback_rank
            entry = fused.setdefault(
                result.doc_id,
                {
                    "score": 0.0,
                    "source_ranks": {},
                    "source_scores": {},
                    "text": result.text,
                    "metadata": dict(result.metadata),
                },
            )
            entry["score"] = float(entry["score"]) + weight / (rrf_k + rank)
            entry["source_ranks"][source_name] = rank  # type: ignore[index]
            entry["source_scores"][source_name] = result.score  # type: ignore[index]
            if not entry["text"] and result.text:
                entry["text"] = result.text
            if not entry["metadata"] and result.metadata:
                entry["metadata"] = dict(result.metadata)

    ordered = sorted(fused.items(), key=lambda item: (-float(item[1]["score"]), item[0]))
    return [
        FusedResult(
            doc_id=doc_id,
            rrf_score=float(entry["score"]),
            rank=rank,
            source_ranks=dict(entry["source_ranks"]),  # type: ignore[arg-type]
            source_scores=dict(entry["source_scores"]),  # type: ignore[arg-type]
            text=str(entry["text"]),
            metadata=dict(entry["metadata"]),  # type: ignore[arg-type]
        )
        for rank, (doc_id, entry) in enumerate(ordered, start=1)
    ]


def _resolve_weights(
    names: Sequence[str],
    weights: Sequence[float] | Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return dict.fromkeys(names, 1.0)
    if isinstance(weights, Mapping):
        return {name: float(weights.get(name, 1.0)) for name in names}
    if len(weights) != len(names):
        raise ValueError("weights must match the number of ranked lists")
    return {name: float(weight) for name, weight in zip(names, weights, strict=True)}
