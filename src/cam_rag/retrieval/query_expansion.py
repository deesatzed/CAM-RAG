"""Dependency-free query expansion helpers for retrieval results."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

DEFAULT_EXPANSION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


def tokenize_expansion_text(text: str) -> list[str]:
    """Tokenize expansion text using the retrieval package's lightweight term shape."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def filter_expansion_terms(
    terms: Iterable[str],
    query: str = "",
    *,
    tokenizer: Callable[[str], list[str]] | None = None,
    stopwords: Iterable[str] | None = None,
    min_term_length: int = 3,
) -> list[str]:
    """Return candidate expansion terms after removing query terms and stopwords."""

    tokenize = tokenizer or tokenize_expansion_text
    blocked = set(tokenize(query))
    blocked.update(
        DEFAULT_EXPANSION_STOPWORDS if stopwords is None else _normalize_terms(stopwords)
    )

    filtered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.lower()
        if (
            len(normalized) < min_term_length
            or normalized in blocked
            or normalized in seen
            or not any(character.isalpha() for character in normalized)
        ):
            continue
        seen.add(normalized)
        filtered.append(normalized)
    return filtered


def extract_expansion_terms(
    items: Iterable[Any],
    query: str = "",
    *,
    tokenizer: Callable[[str], list[str]] | None = None,
    top_k: int = 3,
    max_terms: int = 8,
    stopwords: Iterable[str] | None = None,
    min_term_length: int = 3,
) -> list[str]:
    """Extract ranked expansion terms from top retrieval or evidence-like items."""

    if top_k <= 0 or max_terms <= 0:
        return []

    counts: Counter[str] = Counter()
    first_seen: dict[str, tuple[int, int]] = {}
    tokenize = tokenizer or tokenize_expansion_text
    for item_index, item in enumerate(_take(items, top_k)):
        for term_index, term in enumerate(tokenize(_item_text(item))):
            if term not in first_seen:
                first_seen[term] = (item_index, term_index)
            counts[term] += 1

    ranked_terms = sorted(
        counts,
        key=lambda term: (-counts[term], first_seen[term][0], first_seen[term][1], term),
    )
    return filter_expansion_terms(
        ranked_terms,
        query,
        tokenizer=tokenizer,
        stopwords=stopwords,
        min_term_length=min_term_length,
    )[:max_terms]


def build_expanded_query(
    query: str,
    items: Iterable[Any],
    *,
    tokenizer: Callable[[str], list[str]] | None = None,
    top_k: int = 3,
    max_terms: int = 8,
    stopwords: Iterable[str] | None = None,
    min_term_length: int = 3,
    separator: str = " ",
) -> str:
    """Append extracted expansion terms to the original query string."""

    expansion_terms = extract_expansion_terms(
        items,
        query,
        tokenizer=tokenizer,
        top_k=top_k,
        max_terms=max_terms,
        stopwords=stopwords,
        min_term_length=min_term_length,
    )
    pieces = [query.strip(), *expansion_terms]
    return separator.join(piece for piece in pieces if piece)


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        return _mapping_text(item)

    text = getattr(item, "text", None)
    if text is not None:
        return str(text)

    chunk = getattr(item, "chunk", None)
    if chunk is not None:
        chunk_text = _chunk_text(chunk)
        if chunk_text:
            return chunk_text

    return ""


def _mapping_text(item: Mapping[str, Any]) -> str:
    text = item.get("text")
    if text is not None:
        return str(text)
    chunk = item.get("chunk")
    if chunk is not None:
        return _chunk_text(chunk)
    return ""


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        text = chunk.get("text")
    else:
        text = getattr(chunk, "text", None)
    return "" if text is None else str(text)


def _normalize_terms(terms: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for term in terms:
        normalized.update(tokenize_expansion_text(term))
    return normalized


def _take(items: Iterable[Any], limit: int) -> Sequence[Any]:
    taken: list[Any] = []
    for item in items:
        taken.append(item)
        if len(taken) >= limit:
            break
    return taken
