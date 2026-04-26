"""Dependency-free BM25-style sparse lexical retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult

Tokenizer = Callable[[str], list[str]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight lexical retrieval."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class SparseBM25Retriever:
    """In-memory BM25 retriever with no third-party runtime dependencies."""

    def __init__(
        self,
        documents: Iterable[RetrievalDocument | Mapping[str, object] | tuple[str, str]],
        *,
        tokenizer: Tokenizer = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents = [_coerce_document(document) for document in documents]
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []
        self._avg_document_length = 0.0
        self._build()

    def retrieve(self, query: str, k: int = 20) -> list[RetrievalResult]:
        """Return the top-k documents sorted by BM25 score descending."""

        if k <= 0 or not self.documents:
            return []
        query_terms = self.tokenizer(query)
        if not query_terms:
            return []

        query_counts = Counter(query_terms)
        scored: list[tuple[int, float]] = []
        for index, term_frequencies in enumerate(self._term_frequencies):
            score = self._score(query_counts, term_frequencies, self._document_lengths[index])
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], self.documents[item[0]].doc_id))
        return [
            RetrievalResult(
                doc_id=self.documents[index].doc_id,
                text=self.documents[index].text,
                score=score,
                rank=rank,
                metadata=dict(self.documents[index].metadata),
            )
            for rank, (index, score) in enumerate(scored[:k], start=1)
        ]

    def _build(self) -> None:
        for document in self.documents:
            frequencies = Counter(self.tokenizer(document.text))
            self._term_frequencies.append(frequencies)
            self._document_lengths.append(sum(frequencies.values()))
            self._document_frequencies.update(frequencies.keys())

        if self._document_lengths:
            self._avg_document_length = sum(self._document_lengths) / len(self._document_lengths)

    def _score(
        self,
        query_counts: Counter[str],
        term_frequencies: Counter[str],
        document_length: int,
    ) -> float:
        if document_length == 0 or self._avg_document_length == 0:
            return 0.0

        score = 0.0
        total_documents = len(self.documents)
        for term, query_frequency in query_counts.items():
            term_frequency = term_frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            idf = math.log(
                1
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._avg_document_length
            )
            score += query_frequency * idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


def bm25_search(
    documents: Sequence[RetrievalDocument | Mapping[str, object] | tuple[str, str]],
    query: str,
    *,
    k: int = 20,
    tokenizer: Tokenizer = tokenize,
) -> list[RetrievalResult]:
    """Convenience one-shot BM25 search over a small document collection."""

    return SparseBM25Retriever(documents, tokenizer=tokenizer).retrieve(query, k=k)


def _coerce_document(
    document: RetrievalDocument | Mapping[str, object] | tuple[str, str],
) -> RetrievalDocument:
    if isinstance(document, RetrievalDocument):
        return document
    if isinstance(document, tuple):
        doc_id, text = document
        return RetrievalDocument(doc_id=str(doc_id), text=str(text))
    metadata = document.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError("document metadata must be a dict when provided")
    return RetrievalDocument(
        doc_id=str(document["doc_id"]),
        text=str(document["text"]),
        metadata=metadata,
    )
