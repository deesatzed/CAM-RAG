"""Accuracy contract classification and selective storage filtering.

Contracts classify chunks into hard/medium/soft tiers based on their MoE
accuracy_criticality score.  Selective filtering adjusts evidence scores
based on the chunk's MoE store score to boost high-value content and
penalise boilerplate.  No LLM calls are involved.
"""

from __future__ import annotations

from typing import Literal

from cam_rag.rag.models import Chunk, Evidence

ContractTier = Literal["hard", "medium", "soft"]
ConceptType = Literal[
    "context_sovereignty_rule", "methodology", "frontier_concept", "operational_hint"
]
EmergenceRole = Literal["constrain", "route", "seed", "support"]


def _get_moe_votes(chunk: Chunk) -> dict[str, float]:
    """Extract MoE votes from chunk metadata, returning empty dict if absent."""
    return chunk.metadata.get("moe_votes", {})


def _infer_concept_type(votes: dict[str, float]) -> ConceptType:
    """Infer concept type from MoE vote profile (first match wins)."""
    if votes.get("context_sovereignty", 0.0) >= 0.45:
        return "context_sovereignty_rule"
    if votes.get("methodology", 0.0) >= 0.5:
        return "methodology"
    if votes.get("novelty", 0.0) >= 0.5 or votes.get("emergence_leverage", 0.0) >= 0.5:
        return "frontier_concept"
    return "operational_hint"


def _infer_emergence_role(votes: dict[str, float]) -> EmergenceRole:
    """Infer emergence role from MoE vote profile (first match wins)."""
    if votes.get("context_sovereignty", 0.0) >= 0.45:
        return "constrain"
    if votes.get("methodology", 0.0) >= 0.5:
        return "route"
    if votes.get("novelty", 0.0) >= 0.5 or votes.get("emergence_leverage", 0.0) >= 0.5:
        return "seed"
    return "support"


def classify_contract(chunk: Chunk) -> ContractTier:
    """Classify a chunk's accuracy contract tier.

    Side-effects: sets chunk.metadata["accuracy_contract"],
    chunk.metadata["concept_type"], chunk.metadata["emergence_role"].
    Returns the tier.
    """
    votes = _get_moe_votes(chunk)
    accuracy = votes.get("accuracy_criticality", 0.0)

    if accuracy >= 0.6:
        tier: ContractTier = "hard"
    elif accuracy >= 0.35:
        tier = "medium"
    else:
        tier = "soft"

    chunk.metadata["accuracy_contract"] = tier
    chunk.metadata["concept_type"] = _infer_concept_type(votes)
    chunk.metadata["emergence_role"] = _infer_emergence_role(votes)

    return tier


def classify_contracts(chunks: list[Chunk]) -> list[Chunk]:
    """Classify all chunks. Returns same list (mutated)."""
    for chunk in chunks:
        classify_contract(chunk)
    return chunks


def apply_selective_filter(
    evidence: list[Evidence],
    *,
    threshold: float = 0.55,
) -> list[Evidence]:
    """Adjust evidence scores based on MoE store scores.

    Chunks above threshold: multiplier = 1.0 + (store_score - threshold) * 1.0
    (capped at 1.5).
    Chunks below threshold: multiplier = 0.5 + (store_score / threshold) * 0.5
    (so at threshold it equals 1.0, at 0 it equals 0.5).

    Hard override: if accuracy_criticality >= 0.82 AND reproducibility < 0.70,
    always use max multiplier 1.5.

    After applying multipliers:
    - Re-sort evidence by adjusted score (descending)
    - Update ranks (0-indexed)
    - Store original score in signals["pre_filter_score"]

    Returns the same evidence list (mutated and re-sorted).
    """
    for ev in evidence:
        # Preserve original score
        ev.signals["pre_filter_score"] = ev.score  # type: ignore[typeddict-unknown-key]

        store_score = ev.chunk.metadata.get("moe_store_score", 0.0)
        votes = ev.chunk.metadata.get("moe_votes", {})
        accuracy = votes.get("accuracy_criticality", 0.0)
        reproducibility = votes.get("reproducibility", 0.0)

        # Hard override check
        if accuracy >= 0.82 and reproducibility < 0.70:
            multiplier = 1.5
        elif store_score >= threshold:
            multiplier = 1.0 + (store_score - threshold) * 1.0
            multiplier = min(multiplier, 1.5)
        else:
            if threshold > 0:
                multiplier = 0.5 + (store_score / threshold) * 0.5
            else:
                multiplier = 1.0

        ev.score = ev.score * multiplier

    # Re-sort by adjusted score descending
    evidence.sort(key=lambda e: e.score, reverse=True)

    # Update ranks (0-indexed)
    for i, ev in enumerate(evidence):
        ev.rank = i

    return evidence
