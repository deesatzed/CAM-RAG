"""Chunk-level complexity scoring using text heuristics.

Each chunk is assigned a complexity label ("low", "medium", "high") and a
raw composite score (0.0 - 1.0) based purely on surface-level text analysis.
No LLM calls are involved.  The scores are stored in ``chunk.metadata`` so
that downstream pipeline steps can route chunks to different retrieval or
generation strategies.
"""

from __future__ import annotations

import re
import string
from typing import Literal

from cam_rag.rag.models import Chunk

# ---------------------------------------------------------------------------
# Compiled patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Matches sentence-ending punctuation followed by whitespace or end-of-string
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Technical terms: alphanumeric with internal caps (e.g. SpO2, GCS, pH),
# digits mixed with letters (e.g. PaCO2, 5-HT3), or greek-letter prefixes.
_TECHNICAL_TERM = re.compile(
    r"\b(?:"
    r"[a-z]+[A-Z][a-zA-Z0-9]*"  # camelCase-like: e.g. SpO2, pH
    r"|[A-Z][a-z]+[A-Z][a-zA-Z0-9]*"  # PascalCase mid-cap: e.g. PaCO2
    r"|[A-Za-z]+\d+[A-Za-z]*"  # letters then digits then optional letters: e.g. H2O, 5HT3
    r"|[0-9]+[A-Za-z]+[0-9A-Za-z]*"  # digits then letters: e.g. 5-HT3
    r")\b"
)

# Citation patterns: [1], [12], (Smith 2020), (Smith et al., 2019)
_CITATION = re.compile(
    r"\[\d+\]"
    r"|\([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*,?\s*\d{4}\)"
)

# Bullet / numbered list items
_LIST_ITEM = re.compile(
    r"(?:^|\n)\s*(?:[-*+]|\d+[.)]\s)"
)

ComplexityLabel = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Individual heuristic scorers (each returns 0.0 - 1.0)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split text into word tokens, stripping punctuation."""
    return [
        w.strip(string.punctuation)
        for w in text.split()
        if w.strip(string.punctuation)
    ]


def _sentence_count_score(sentences: list[str]) -> float:
    """More sentences in a chunk suggests structural complexity.

    Scale: 1 sentence -> 0.0, >=10 sentences -> 1.0 (linear).
    """
    n = len(sentences)
    if n <= 1:
        return 0.0
    return min((n - 1) / 9.0, 1.0)


def _avg_sentence_length_score(sentences: list[str]) -> float:
    """Longer sentences tend to express more complex ideas.

    Scale: <=8 words avg -> 0.0, >=30 words avg -> 1.0 (linear).
    """
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    if avg <= 8:
        return 0.0
    return min((avg - 8) / 22.0, 1.0)


def _vocabulary_diversity_score(tokens: list[str]) -> float:
    """Higher type/token ratio implies more diverse vocabulary.

    Scale: 0.0 at TTR<=0.3, 1.0 at TTR>=0.9 (linear).
    """
    if not tokens:
        return 0.0
    lower_tokens = [t.lower() for t in tokens]
    ttr = len(set(lower_tokens)) / len(lower_tokens)
    if ttr <= 0.3:
        return 0.0
    return min((ttr - 0.3) / 0.6, 1.0)


def _technical_term_density_score(tokens: list[str], text: str) -> float:
    """Density of technical / mixed-case / long terms.

    Combines regex-matched technical patterns with long-word counts.
    Scale: 0% -> 0.0, >=8% density -> 1.0.
    """
    if not tokens:
        return 0.0
    tech_count = len(_TECHNICAL_TERM.findall(text))
    long_count = sum(1 for t in tokens if len(t) > 12)
    density = (tech_count + long_count) / len(tokens)
    return min(density / 0.08, 1.0)


def _citation_density_score(text: str, token_count: int) -> float:
    """Density of academic/medical citation patterns.

    Scale: 0 citations -> 0.0, >=3 per 100 tokens -> 1.0.
    """
    if token_count == 0:
        return 0.0
    cites = len(_CITATION.findall(text))
    density = cites / token_count
    return min(density / 0.03, 1.0)


def _list_density_score(text: str, token_count: int) -> float:
    """Density of bulleted / numbered list items.

    Structured lists indicate organized, potentially complex content.
    Scale: 0 items -> 0.0, >=5 per 100 tokens -> 1.0.
    """
    if token_count == 0:
        return 0.0
    items = len(_LIST_ITEM.findall(text))
    density = items / token_count
    return min(density / 0.05, 1.0)


def _numeric_density_score(tokens: list[str]) -> float:
    """Ratio of tokens containing digits (measurements, dosages, etc.).

    Scale: 0% -> 0.0, >=20% -> 1.0.
    """
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if any(c.isdigit() for c in t))
    density = numeric / len(tokens)
    return min(density / 0.20, 1.0)


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

# Weights for each heuristic (must sum to 1.0)
_WEIGHTS: dict[str, float] = {
    "sentence_count": 0.10,
    "avg_sentence_length": 0.15,
    "vocabulary_diversity": 0.15,
    "technical_term_density": 0.20,
    "citation_density": 0.10,
    "list_density": 0.10,
    "numeric_density": 0.20,
}


def _composite_score(text: str) -> float:
    """Return a composite complexity score in [0.0, 1.0]."""
    if not text or not text.strip():
        return 0.0

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        # No sentence-ending punctuation: treat entire text as one sentence
        sentences = [text.strip()]

    tokens = _tokenize(text)
    token_count = len(tokens)

    scores = {
        "sentence_count": _sentence_count_score(sentences),
        "avg_sentence_length": _avg_sentence_length_score(sentences),
        "vocabulary_diversity": _vocabulary_diversity_score(tokens),
        "technical_term_density": _technical_term_density_score(tokens, text),
        "citation_density": _citation_density_score(text, token_count),
        "list_density": _list_density_score(text, token_count),
        "numeric_density": _numeric_density_score(tokens),
    }

    composite = sum(scores[k] * _WEIGHTS[k] for k in _WEIGHTS)
    # Clamp to [0.0, 1.0] for safety
    return max(0.0, min(1.0, composite))


def _label_from_score(score: float) -> ComplexityLabel:
    """Bucket a raw score into a complexity label."""
    if score <= 0.33:
        return "low"
    if score <= 0.66:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_chunk_complexity(chunk: Chunk) -> ComplexityLabel:
    """Score a single chunk's complexity and store results in metadata.

    Returns the complexity label ("low", "medium", "high").
    Side-effect: sets ``chunk.metadata["complexity"]`` and
    ``chunk.metadata["complexity_score"]``.
    """
    raw = _composite_score(chunk.text)
    label = _label_from_score(raw)
    chunk.metadata["complexity"] = label
    chunk.metadata["complexity_score"] = round(raw, 4)
    return label


def score_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Score every chunk in a list and return the same list (mutated).

    Each chunk gets ``metadata["complexity"]`` (label) and
    ``metadata["complexity_score"]`` (float).
    """
    for chunk in chunks:
        score_chunk_complexity(chunk)
    return chunks
