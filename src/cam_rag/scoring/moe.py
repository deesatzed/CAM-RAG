"""Mixture-of-Experts importance scoring for retrieval chunks.

Seven domain experts independently score each chunk based on keyword
presence.  The per-expert scores are combined via a weighted average
(``combined_score``) and a separate linear formula (``store_score``) that
emphasises accuracy-critical and novel content while penalising
boilerplate.

No LLM calls are involved -- scoring is pure heuristic text analysis so
it can run at ingestion time without network round-trips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cam_rag.rag.models import Chunk

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from *text*."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Expert definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _ExpertDef:
    """Immutable definition of a single MoE expert."""

    name: str
    weight: float
    keywords: frozenset[str]


_EXPERTS: tuple[_ExpertDef, ...] = (
    _ExpertDef(
        name="accuracy_criticality",
        weight=1.25,
        keywords=frozenset({
            "must", "never", "safety", "security", "compliance",
            "critical", "contraindicated", "warning",
        }),
    ),
    _ExpertDef(
        name="novelty",
        weight=1.15,
        keywords=frozenset({
            "novel", "alien", "frontier", "first-principles",
            "unprecedented", "breakthrough",
        }),
    ),
    _ExpertDef(
        name="methodology",
        weight=1.10,
        keywords=frozenset({
            "protocol", "algorithm", "schema", "gate", "workflow",
            "pipeline", "procedure",
        }),
    ),
    _ExpertDef(
        name="context_sovereignty",
        weight=1.15,
        keywords=frozenset({
            "context", "evidence", "grounded", "citation",
            "reference", "verified", "provenance",
        }),
    ),
    _ExpertDef(
        name="reproducibility",
        weight=1.00,
        keywords=frozenset({
            "import", "basic", "boilerplate", "template", "default",
            "standard", "trivial", "generic",
        }),
    ),
    _ExpertDef(
        name="emergence_leverage",
        weight=1.25,
        keywords=frozenset({
            "capability", "emergent", "feedback", "recursive",
            "leverage", "cascade", "synergy",
        }),
    ),
    _ExpertDef(
        name="usability",
        weight=1.10,
        keywords=frozenset({
            "gate", "prompt", "command", "score", "test",
            "validator", "actionable", "executable",
        }),
    ),
)

_SATURATION = 5


def _score_expert(expert: _ExpertDef, tokens: list[str]) -> tuple[float, int]:
    """Return (score, hits) for a single expert against *tokens*.

    Score is ``min(1.0, hits / _SATURATION)`` where *hits* is the number
    of tokens that appear in the expert's keyword set.
    """
    hits = sum(1 for t in tokens if t in expert.keywords)
    score = min(1.0, hits / _SATURATION)
    return score, hits


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExpertVote:
    """Immutable record of one expert's evaluation of a chunk."""

    name: str
    weight: float
    score: float
    hits: int


@dataclass(frozen=True, slots=True)
class MoEResult:
    """Aggregated MoE scoring result for a single chunk."""

    combined_score: float
    store_score: float
    votes: dict[str, ExpertVote]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary for JSON / metadata storage."""
        return {
            "combined_score": self.combined_score,
            "store_score": self.store_score,
            "votes": {name: vote.score for name, vote in self.votes.items()},
        }


# ---------------------------------------------------------------------------
# Store-score coefficients
# ---------------------------------------------------------------------------

_STORE_COEFFICIENTS: dict[str, float] = {
    "accuracy_criticality": 0.25,
    "novelty": 0.20,
    "methodology": 0.20,
    "emergence_leverage": 0.15,
    "context_sovereignty": 0.10,
    "reproducibility": -0.30,
}

_COMPLEXITY_COEFFICIENT = 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_chunk_moe(chunk: Chunk) -> MoEResult:
    """Score a single chunk using the 7-expert MoE panel.

    Side-effects -- the following keys are written into ``chunk.metadata``:

    * ``moe_combined_score`` -- weighted average across all experts
    * ``moe_store_score``    -- linear formula for long-term storage ranking
    * ``moe_votes``          -- ``dict[str, float]`` of per-expert scores

    Returns the full :class:`MoEResult` for programmatic consumption.
    """
    tokens = _tokenize(chunk.text)

    # -- Collect votes -------------------------------------------------------
    votes: dict[str, ExpertVote] = {}
    for expert in _EXPERTS:
        score, hits = _score_expert(expert, tokens)
        votes[expert.name] = ExpertVote(
            name=expert.name,
            weight=expert.weight,
            score=score,
            hits=hits,
        )

    # -- Combined score (weighted average) -----------------------------------
    total_weight = sum(v.weight for v in votes.values())
    combined = (
        sum(v.weight * v.score for v in votes.values()) / total_weight
        if total_weight > 0
        else 0.0
    )

    # -- Store score (linear formula with complexity) ------------------------
    complexity_score: float = chunk.metadata.get("complexity_score", 0.0)

    store = sum(
        coeff * votes[name].score
        for name, coeff in _STORE_COEFFICIENTS.items()
    ) + _COMPLEXITY_COEFFICIENT * complexity_score

    result = MoEResult(
        combined_score=round(combined, 6),
        store_score=round(store, 6),
        votes=votes,
    )

    # -- Side-effects on chunk metadata --------------------------------------
    chunk.metadata["moe_combined_score"] = result.combined_score
    chunk.metadata["moe_store_score"] = result.store_score
    chunk.metadata["moe_votes"] = {
        name: vote.score for name, vote in votes.items()
    }

    return result


def score_chunks_moe(chunks: list[Chunk]) -> list[Chunk]:
    """Score every chunk in *chunks* and return the same list (mutated).

    Each chunk gets ``metadata["moe_combined_score"]``,
    ``metadata["moe_store_score"]``, and ``metadata["moe_votes"]`` set by
    :func:`score_chunk_moe`.
    """
    for chunk in chunks:
        score_chunk_moe(chunk)
    return chunks
