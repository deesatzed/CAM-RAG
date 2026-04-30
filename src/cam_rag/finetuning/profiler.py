"""Corpus profiling and domain gap scoring for embedding fine-tuning.

Analyzes a corpus to determine whether the base embedding model's
vocabulary and representation space are well-suited for the domain.
A high gap score indicates that fine-tuning is likely to improve
retrieval quality.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from cam_rag.documents.complexity import _TECHNICAL_TERM
from cam_rag.rag.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusProfile:
    """Statistical profile of a corpus relative to an embedding model."""

    oov_ratio: float
    avg_subword_fragmentation: float
    entity_density: float
    embedding_dispersion: float
    vocabulary_overlap: float
    num_chunks_sampled: int
    total_tokens_sampled: int


@dataclass(frozen=True)
class DomainGapReport:
    """Result of domain gap analysis."""

    gap_score: float
    recommendation: str
    profile: CorpusProfile
    signal_weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default signal weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "oov_ratio": 0.25,
    "avg_subword_fragmentation": 0.25,
    "entity_density": 0.15,
    "embedding_dispersion": 0.20,
    "vocabulary_overlap": 0.15,
}


# ---------------------------------------------------------------------------
# CorpusProfiler
# ---------------------------------------------------------------------------


class CorpusProfiler:
    """Analyze a corpus against an embedding model's tokenizer and representation.

    Parameters
    ----------
    model:
        A ``LocalSentenceTransformerBackend`` (or any object with a
        ``_model`` attribute exposing a sentence-transformers model and
        ``embed`` / ``embed_batch`` methods).
    sample_size:
        Maximum number of chunks to sample for analysis. Set to 0
        to use all chunks.
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        model: Any,
        *,
        sample_size: int = 200,
        seed: int = 42,
    ) -> None:
        self._model = model
        self._sample_size = sample_size
        self._seed = seed
        self._tokenizer = self._get_tokenizer()

    def profile(self, chunks: list[Chunk]) -> CorpusProfile:
        """Compute a statistical profile of *chunks* relative to the model."""
        sample = self._sample(chunks)
        if not sample:
            return CorpusProfile(
                oov_ratio=0.0,
                avg_subword_fragmentation=0.0,
                entity_density=0.0,
                embedding_dispersion=0.0,
                vocabulary_overlap=0.0,
                num_chunks_sampled=0,
                total_tokens_sampled=0,
            )

        texts = [c.text for c in sample]
        all_whitespace_tokens: list[str] = []
        for text in texts:
            all_whitespace_tokens.extend(text.split())

        oov_ratio = self._compute_oov_ratio(all_whitespace_tokens)
        avg_frag = self._compute_avg_fragmentation(all_whitespace_tokens)
        entity_density = self._compute_entity_density(texts)
        embedding_disp = self._compute_embedding_dispersion(sample)
        vocab_overlap = self._compute_vocabulary_overlap(all_whitespace_tokens)

        return CorpusProfile(
            oov_ratio=oov_ratio,
            avg_subword_fragmentation=avg_frag,
            entity_density=entity_density,
            embedding_dispersion=embedding_disp,
            vocabulary_overlap=vocab_overlap,
            num_chunks_sampled=len(sample),
            total_tokens_sampled=len(all_whitespace_tokens),
        )

    # ------------------------------------------------------------------
    # Signal computations
    # ------------------------------------------------------------------

    def _compute_oov_ratio(self, tokens: list[str]) -> float:
        """Fraction of whitespace tokens that fragment into 3+ subword pieces."""
        if not tokens:
            return 0.0
        oov_count = 0
        for token in tokens:
            pieces = self._tokenize_word(token)
            if len(pieces) >= 3:
                oov_count += 1
        return oov_count / len(tokens)

    def _compute_avg_fragmentation(self, tokens: list[str]) -> float:
        """Mean number of subword pieces per whitespace token."""
        if not tokens:
            return 0.0
        total_pieces = 0
        for token in tokens:
            pieces = self._tokenize_word(token)
            total_pieces += len(pieces)
        return total_pieces / len(tokens)

    def _compute_entity_density(self, texts: list[str]) -> float:
        """Fraction of whitespace tokens matching technical-term patterns.

        Reuses ``_TECHNICAL_TERM`` from ``cam_rag.documents.complexity``.
        """
        all_text = " ".join(texts)
        tokens = all_text.split()
        if not tokens:
            return 0.0
        tech_count = len(_TECHNICAL_TERM.findall(all_text))
        return tech_count / len(tokens)

    def _compute_embedding_dispersion(self, chunks: list[Chunk]) -> float:
        """1 - mean pairwise cosine similarity over a chunk sample.

        Low mean similarity (high dispersion) means the model *can*
        distinguish domain concepts. We return ``1 - mean_sim`` so
        that *low* dispersion (model struggles) produces a *high* gap signal.
        """
        # Use a small sub-sample for pairwise computation
        sub = chunks[: min(50, len(chunks))]
        if len(sub) < 2:
            return 0.0

        texts = [c.text for c in sub]
        if hasattr(self._model, "embed_batch"):
            vectors = self._model.embed_batch(texts)
        else:
            vectors = [self._model.embed(t) for t in texts]

        sims: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(_cosine_sim(vectors[i], vectors[j]))

        if not sims:
            return 0.0
        mean_sim = sum(sims) / len(sims)
        # Invert: high mean similarity = model can't distinguish = high gap
        return mean_sim

    def _compute_vocabulary_overlap(self, tokens: list[str]) -> float:
        """Fraction of corpus tokens present as single pieces in model vocab.

        High overlap means the model already knows the vocabulary well,
        so we return ``1 - overlap`` as the gap signal.
        """
        if not tokens:
            return 0.0
        in_vocab = 0
        for token in tokens:
            pieces = self._tokenize_word(token)
            if len(pieces) == 1:
                in_vocab += 1
        # Return the *gap* signal: 1 - overlap
        return 1.0 - (in_vocab / len(tokens))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self) -> Any:
        """Extract the word-piece tokenizer from the sentence-transformer model."""
        st_model = self._model._model if hasattr(self._model, "_model") else self._model
        # sentence-transformers wraps a HF tokenizer
        if hasattr(st_model, "tokenizer"):
            return st_model.tokenizer
        # Fallback: try first module's tokenizer
        if hasattr(st_model, "_modules"):
            for module in st_model._modules.values():
                if hasattr(module, "tokenizer"):
                    return module.tokenizer
        raise ValueError(
            "Could not find a tokenizer on the model. "
            "Ensure you pass a LocalSentenceTransformerBackend or SentenceTransformer."
        )

    def _tokenize_word(self, word: str) -> list[int]:
        """Tokenize a single whitespace token into subword piece IDs."""
        return self._tokenizer.encode(word, add_special_tokens=False)

    def _sample(self, chunks: list[Chunk]) -> list[Chunk]:
        """Sample up to ``sample_size`` chunks reproducibly."""
        if self._sample_size <= 0 or len(chunks) <= self._sample_size:
            return list(chunks)
        rng = random.Random(self._seed)
        return rng.sample(chunks, self._sample_size)


# ---------------------------------------------------------------------------
# DomainGapScorer
# ---------------------------------------------------------------------------


class DomainGapScorer:
    """Combine corpus profile signals into a 0-1 domain gap score.

    Parameters
    ----------
    weights:
        Signal name -> weight mapping. Weights are normalized to sum to 1.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, profile: CorpusProfile) -> DomainGapReport:
        """Produce a gap report from a corpus profile."""
        signals = {
            "oov_ratio": _clamp(profile.oov_ratio / 0.4),  # 40% OOV → 1.0
            "avg_subword_fragmentation": _clamp(
                (profile.avg_subword_fragmentation - 1.3) / 2.2
            ),  # 1.3 baseline → 0, 3.5 → 1.0
            "entity_density": _clamp(profile.entity_density / 0.08),  # 8% → 1.0
            "embedding_dispersion": _clamp(profile.embedding_dispersion),
            "vocabulary_overlap": _clamp(profile.vocabulary_overlap / 0.4),  # 40% not in vocab → 1.0
        }

        gap = sum(signals.get(k, 0.0) * w for k, w in self.weights.items())
        gap = _clamp(gap)

        if gap < 0.3:
            recommendation = "Base model adequate, marginal gains expected"
        elif gap < 0.6:
            recommendation = "Moderate gap, fine-tuning recommended"
        else:
            recommendation = "Significant gap, fine-tuning strongly recommended"

        return DomainGapReport(
            gap_score=round(gap, 4),
            recommendation=recommendation,
            profile=profile,
            signal_weights=dict(self.weights),
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
