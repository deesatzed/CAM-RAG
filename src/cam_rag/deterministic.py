"""Deterministic retrieval backend.

This module provides a compliance-friendly retrieval path for regulated domains,
replacing vector embeddings with reproducible TF-IDF scoring, metadata aliases,
and configurable routing rules.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import yaml


@dataclass(frozen=True)
class Document:
    """A document to be indexed and retrieved."""

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    domain: str = "default"


@dataclass
class RetrievalResult:
    """Result of a deterministic retrieval query."""

    document: Document
    score: float
    routing_reason: str | None = None


class DeterministicRetriever:
    """
    A deterministic retrieval backend using TF-IDF with metadata aliases and
    explicit disambiguation rules.

    Features:
    - TF-IDF scoring for reproducible, explainable results.
    - Metadata alias matching (e.g., 'bug' -> 'issue type: bug').
    - Domain-specific routing rules configurable via YAML.
    - No neural embeddings or external API dependencies.

    Usage::

        retriever = DeterministicRetriever()
        retriever.index_documents([Document(id="1", text="...", metadata={"type": "bug"})])
        results = retriever.retrieve("login error", top_k=5)
    """

    def __init__(self, rules_path: str | None = None):
        self._documents: dict[str, Document] = {}
        self._doc_freq: Counter[str] = Counter()
        self._total_docs: int = 0
        self._aliases: defaultdict[str, set[str]] = defaultdict(set)
        self._routing_rules: list[dict] = []

        if rules_path:
            self._load_rules(rules_path)

    # Public API

    def index_documents(self, documents: Sequence[Document]) -> None:
        """Index a sequence of documents for retrieval.

        This operation is idempotent per document ID; the last document
        with a given ID wins. Updates internal term frequencies for
        reproducible TF-IDF scoring.

        Args:
            documents: An iterable of Document objects to index.
        """
        for doc in documents:
            self._documents[doc.id] = doc
        self._rebuild_document_frequencies()
        self._total_docs = len(self._documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve documents matching the query using deterministic scoring.

        Scoring combines:
        1. TF-IDF cosine similarity between query tokens and document tokens.
        2. Metadata alias matches (e.g., query contains "bug" and document
           has metadata type=bug).
        3. Domain-specific routing rules applied as score multipliers.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return (default 5).
            domain: Optional domain filter (e.g., "support", "legal").
                If provided, only documents in that domain are scored.

        Returns:
            A list of RetrievalResult objects, sorted by descending score.
        """
        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_terms = set(query_tokens)

        candidates = list(self._documents.values())
        if domain:
            candidates = [d for d in candidates if d.domain == domain]

        results: list[RetrievalResult] = []
        for doc in candidates:
            score = self._compute_tfidf(doc, query_tf, query_terms)
            score += self._metadata_alias_score(doc, query_terms)
            routing_reason = None
            matched_rule = self._apply_routing_rules(doc, query)
            if matched_rule:
                score *= matched_rule.get("boost", 1.5)
                routing_reason = matched_rule.get("reason", "routing rule applied")
            if score > 0.0:
                results.append(RetrievalResult(doc, score, routing_reason))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def add_alias(self, alias: str, targets: set[str]) -> None:
        """Register a metadata alias.

        When a query contains the alias term, documents whose metadata
        keys or values match any of the targets receive a score boost.

        Args:
            alias: The query term to match.
            targets: Set of metadata key-value pairs (formatted as
                "key:value" or just "key") to boost.
        """
        self._aliases[alias].update(targets)

    def get_stats(self) -> dict[str, int]:
        """Return basic statistics about the index."""
        return {
            "documents": self._total_docs,
            "terms": len(self._doc_freq),
        }

    # Internal: rule loading

    def _load_rules(self, path: str) -> None:
        """Load routing rules from a YAML file.

        Expected YAML structure::

            aliases:
              bug: ["type:bug", "category:defect"]
              legal: ["domain:compliance"]

            routing:
              - domain: support
                keywords: ["login", "password", "error"]
                boost: 2.0
                reason: "high-priority support keyword"
              - domain: legal
                keywords: ["gdpr", "compliance"]
                boost: 1.8
                reason: "regulatory requirement"
        """
        with open(path) as f:
            rules = yaml.safe_load(f)

        if not rules:
            return

        # Load aliases
        raw_aliases = rules.get("aliases", {})
        for alias, targets in raw_aliases.items():
            self._aliases[alias].update(targets)

        # Load routing rules
        self._routing_rules = rules.get("routing", [])

    # Internal: scoring

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase tokens, stripping punctuation."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 1]

    def _compute_tfidf(
        self,
        doc: Document,
        query_tf: Counter[str],
        query_terms: set[str],
    ) -> float:
        """Compute TF-IDF cosine similarity between document and query."""
        doc_tokens = self._tokenize(doc.text)
        if not doc_tokens:
            return 0.0

        doc_tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        norm_q = 0.0
        norm_d = 0.0

        query_len = sum(query_tf.values()) or 1
        all_terms = query_terms | set(doc_tf)
        for term in all_terms:
            idf = self._idf(term)
            # query weight
            qw = (query_tf.get(term, 0) / query_len) * idf
            # doc weight
            dw = (doc_tf.get(term, 0) / doc_len) * idf
            score += qw * dw
            norm_q += qw * qw
            norm_d += dw * dw

        norm_q = math.sqrt(norm_q) if norm_q else 1.0
        norm_d = math.sqrt(norm_d) if norm_d else 1.0
        return score / (norm_q * norm_d) if norm_q and norm_d else 0.0

    def _rebuild_document_frequencies(self) -> None:
        """Recompute document frequencies from the current document index."""
        self._doc_freq.clear()
        for doc in self._documents.values():
            for token in set(self._tokenize(doc.text)):
                self._doc_freq[token] += 1

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self._total_docs - df + 0.5) / (df + 0.5))

    def _metadata_alias_score(self, doc: Document, query_terms: set[str]) -> float:
        """Compute score boost from metadata alias matches.

        Returns a score in [0, 1] based on how many query terms match
        registered aliases and how many of those alias targets match
        the document's metadata.
        """
        if not self._aliases:
            return 0.0

        matched_aliases = 0
        total_considerations = 0

        for term in query_terms:
            if term not in self._aliases:
                continue
            total_considerations += 1
            targets = self._aliases[term]
            for meta_key, meta_value in doc.metadata.items():
                if meta_key in targets or f"{meta_key}:{meta_value}" in targets:
                    matched_aliases += 1
                    break

        if total_considerations == 0:
            return 0.0
        return matched_aliases / total_considerations

    def _apply_routing_rules(
        self,
        doc: Document,
        query: str,
    ) -> dict | None:
        """Apply domain-specific routing rules to boost/penalize documents.

        Returns the first matching rule dict, or None if no rule matches.
        """
        query_lower = query.lower()
        for rule in self._routing_rules:
            rule_domain = rule.get("domain", "")
            if doc.domain != rule_domain:
                continue
            keywords = rule.get("keywords", [])
            for kw in keywords:
                if kw.lower() in query_lower:
                    return rule
        return None
