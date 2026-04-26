"""Deterministic lexical utilities for methodology retrieval."""

from __future__ import annotations

import re


STOPWORDS = {
    "also",
    "and",
    "brain",
    "category",
    "from",
    "into",
    "methfam",
    "mined",
    "pattern",
    "python",
    "repofrax",
    "should",
    "source",
    "that",
    "the",
    "this",
    "uses",
    "which",
    "with",
}


def tokens(text: str) -> set[str]:
    result: set[str] = set()
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(token) < 4 or token in STOPWORDS:
            continue
        result.add(token)
    return result


def overlap_score(query: str, document: str) -> float:
    query_tokens = tokens(query)
    document_tokens = tokens(document)
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens.intersection(document_tokens)) / len(query_tokens)
