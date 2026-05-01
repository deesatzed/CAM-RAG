"""Corpus-level statistical signals for adaptive retrieval tuning.

Computes lightweight statistics over a document corpus that inform
pipeline strategy decisions — e.g. whether BM25 is likely to help,
whether vocabulary diversity is high enough for sparse retrieval,
or whether documents are too short for effective chunking.

All computations use stdlib only (no external dependencies).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusSignals:
    """Immutable snapshot of corpus-level statistics.

    Attributes
    ----------
    avg_doc_length:
        Mean token count per document.
    doc_length_std:
        Standard deviation of per-document token counts.
    corpus_size:
        Total number of documents in the corpus.
    vocab_overlap_ratio:
        Fraction of the vocabulary that appears in more than 10% of
        documents.  High values indicate repetitive/boilerplate
        language; low values indicate diverse terminology.
    unique_term_ratio:
        vocabulary_size / total_token_count — a measure of lexical
        diversity.  Higher values mean more unique terms relative to
        corpus length.
    short_doc_fraction:
        Fraction of documents with fewer than 200 tokens.
    """

    avg_doc_length: float
    doc_length_std: float
    corpus_size: int
    vocab_overlap_ratio: float
    unique_term_ratio: float
    short_doc_fraction: float


def compute_corpus_signals(
    docs: list[str],
    tokenizer: Callable[[str], list[str]],
) -> CorpusSignals:
    """Compute corpus-level signals from raw document texts.

    Parameters
    ----------
    docs:
        List of document texts.
    tokenizer:
        A callable that splits a text string into a list of token
        strings.  Any tokenizer that returns ``list[str]`` works
        (e.g. ``str.split``, spaCy, NLTK, etc.).

    Returns
    -------
    CorpusSignals
        Frozen dataclass with computed statistics.
    """
    corpus_size = len(docs)

    if corpus_size == 0:
        return CorpusSignals(
            avg_doc_length=0.0,
            doc_length_std=0.0,
            corpus_size=0,
            vocab_overlap_ratio=0.0,
            unique_term_ratio=0.0,
            short_doc_fraction=0.0,
        )

    # Tokenize every document once
    tokenized_docs: list[list[str]] = [tokenizer(doc) for doc in docs]

    # Per-document token counts
    doc_lengths: list[int] = [len(tokens) for tokens in tokenized_docs]

    # avg_doc_length
    avg_doc_length = sum(doc_lengths) / corpus_size

    # doc_length_std
    if corpus_size > 1:
        doc_length_std = statistics.stdev(doc_lengths)
    else:
        doc_length_std = 0.0

    # Total tokens across the corpus
    total_tokens = sum(doc_lengths)

    # Build document-frequency counter: for each term, how many docs contain it
    doc_freq: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for tokens in tokenized_docs:
        unique_in_doc = set(tokens)
        vocabulary.update(unique_in_doc)
        for term in unique_in_doc:
            doc_freq[term] += 1

    vocab_size = len(vocabulary)

    # vocab_overlap_ratio
    if corpus_size == 1:
        # Single doc: every term appears in 100% of docs
        vocab_overlap_ratio = 1.0
    elif vocab_size == 0:
        vocab_overlap_ratio = 0.0
    else:
        threshold = 0.1 * corpus_size
        common_term_count = sum(
            1 for freq in doc_freq.values() if freq > threshold
        )
        vocab_overlap_ratio = common_term_count / vocab_size

    # unique_term_ratio
    if total_tokens == 0:
        unique_term_ratio = 0.0
    else:
        unique_term_ratio = vocab_size / total_tokens

    # short_doc_fraction
    short_count = sum(1 for length in doc_lengths if length < 200)
    short_doc_fraction = short_count / corpus_size

    return CorpusSignals(
        avg_doc_length=avg_doc_length,
        doc_length_std=doc_length_std,
        corpus_size=corpus_size,
        vocab_overlap_ratio=vocab_overlap_ratio,
        unique_term_ratio=unique_term_ratio,
        short_doc_fraction=short_doc_fraction,
    )
