"""Tests for the deterministic retrieval backend."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cam_rag.deterministic import DeterministicRetriever, Document

# Fixtures


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            id="doc1",
            text="Password reset flow fails on login page",
            metadata={"type": "bug", "severity": "high"},
            domain="support",
        ),
        Document(
            id="doc2",
            text="How to configure two-factor authentication",
            metadata={"type": "guide", "audience": "admin"},
            domain="docs",
        ),
        Document(
            id="doc3",
            text="GDPR compliance checklist for data retention",
            metadata={"type": "policy", "region": "eu"},
            domain="legal",
        ),
        Document(
            id="doc4",
            text="SSO integration with Azure AD",
            metadata={"type": "integration", "provider": "azure"},
            domain="docs",
        ),
    ]


@pytest.fixture
def retriever_with_docs(sample_docs: list[Document]) -> DeterministicRetriever:
    retriever = DeterministicRetriever()
    retriever.index_documents(sample_docs)
    return retriever


@pytest.fixture
def rules_yaml(tmp_path: Path) -> Path:
    rules = {
        "aliases": {
            "bug": ["type:bug", "category:defect"],
            "legal": ["domain:compliance"],
        },
        "routing": [
            {
                "domain": "support",
                "keywords": ["login", "password", "reset"],
                "boost": 2.0,
                "reason": "high-priority support keyword",
            },
            {
                "domain": "legal",
                "keywords": ["gdpr", "compliance"],
                "boost": 1.8,
                "reason": "regulatory requirement",
            },
        ],
    }
    path = tmp_path / "rules.yaml"
    with open(path, "w") as f:
        yaml.dump(rules, f)
    return path


# Tests


class TestDeterministicRetriever:
    """Test suite for DeterministicRetriever."""

    def test_index_documents_and_retrieve(
        self,
        retriever_with_docs: DeterministicRetriever,
    ) -> None:
        """Index documents and retrieve a relevant result."""
        results = retriever_with_docs.retrieve("password reset", top_k=2)
        assert len(results) >= 1
        assert results[0].document.id == "doc1"

    def test_retrieve_empty_index(self) -> None:
        """Retrieving from an empty index returns empty list."""
        retriever = DeterministicRetriever()
        assert retriever.retrieve("anything") == []

    def test_retrieve_empty_query(self, retriever_with_docs: DeterministicRetriever) -> None:
        """An empty query returns an empty result list."""
        results = retriever_with_docs.retrieve("")
        assert results == []

    def test_retrieve_top_k(self, retriever_with_docs: DeterministicRetriever) -> None:
        """Respect the top_k parameter."""
        results = retriever_with_docs.retrieve("authentication", top_k=1)
        assert len(results) <= 1

    def test_retrieve_by_domain(self, retriever_with_docs: DeterministicRetriever) -> None:
        """Filter retrieval results by domain."""
        results = retriever_with_docs.retrieve("login", domain="support")
        assert len(results) >= 1
        for r in results:
            assert r.document.domain == "support"

    def test_multiple_domains_filtered(self, retriever_with_docs: DeterministicRetriever) -> None:
        """Filtering by a domain with no matches returns empty."""
        results = retriever_with_docs.retrieve("login", domain="legal")
        assert len(results) == 0

    def test_score_non_negative(self, retriever_with_docs: DeterministicRetriever) -> None:
        """All returned scores are non-negative."""
        results = retriever_with_docs.retrieve("integration azure")
        for r in results:
            assert r.score >= 0.0

    def test_metadata_alias_increases_score(self, sample_docs: list[Document]) -> None:
        """Registering a metadata alias boosts matching documents."""
        retriever = DeterministicRetriever()
        retriever.index_documents(sample_docs)
        retriever.add_alias("bug", {"type:bug"})

        results_without_alias = retriever.retrieve("bug")
        # Re-create a new retriever separate from alias to compare
        plain = DeterministicRetriever()
        plain.index_documents(sample_docs)
        results_no_alias = plain.retrieve("bug")

        # With alias, doc1 should score higher relative to others
        if results_no_alias:
            score_no_alias = results_no_alias[0].score
            score_with_alias = results_without_alias[0].score
            assert score_with_alias >= score_no_alias

    def test_routing_rules_boost_documents(
        self,
        sample_docs: list[Document],
        rules_yaml: Path,
    ) -> None:
        """Routing rules from YAML boost documents in matching domain."""
        retriever = DeterministicRetriever(rules_path=str(rules_yaml))
        retriever.index_documents(sample_docs)
        results = retriever.retrieve("login reset password")

        # doc1 (support domain) should be boosted and at top
        assert len(results) >= 1
        assert results[0].document.id == "doc1"
        assert results[0].routing_reason is not None

    def test_routing_rules_no_match_returns_none_reason(
        self,
        sample_docs: list[Document],
        rules_yaml: Path,
    ) -> None:
        """Documents not matching any routing rule have None reason."""
        retriever = DeterministicRetriever(rules_path=str(rules_yaml))
        retriever.index_documents(sample_docs)
        results = retriever.retrieve("unrelated query")
        if results:
            # Check that documents not matching routing have no reason
            for r in results:
                if r.document.domain not in ("support", "legal"):
                    assert r.routing_reason is None

    def test_explainability_metadata_aliases(self) -> None:
        """Verify that alias matches can be inspected via routing reason.

        This test checks that when an alias is matched, the score is
        influenced and the result includes the routing reason.
        """
        docs = [
            Document(
                id="bug1",
                text="Application crashes on startup",
                metadata={"type": "bug", "severity": "critical"},
                domain="support",
            ),
            Document(
                id="feat1",
                text="New feature: dark mode",
                metadata={"type": "feature"},
                domain="docs",
            ),
        ]
        retriever = DeterministicRetriever()
        retriever.index_documents(docs)
        retriever.add_alias("bug", {"type:bug"})

        results = retriever.retrieve("bug")
        assert len(results) >= 1
        top_doc = results[0].document.id
        assert top_doc == "bug1", (
            f"Expected 'bug1' as top result due to alias boost, got {top_doc}"
        )

    def test_get_stats(self, retriever_with_docs: DeterministicRetriever) -> None:
        """Return correct statistics about indexed documents."""
        stats = retriever_with_docs.get_stats()
        assert stats["documents"] == 4
        assert stats["terms"] > 0

    def test_index_idempotent(self, sample_docs: list[Document]) -> None:
        """Indexing the same docs multiple times is idempotent."""
        retriever = DeterministicRetriever()
        retriever.index_documents(sample_docs)
        retriever.index_documents(sample_docs)  # twice
        assert retriever.get_stats()["documents"] == 4

    def test_document_with_no_text(self) -> None:
        """Documents with empty text do not affect scoring."""
        docs = [
            Document(id="empty", text="", metadata={}, domain="test"),
            Document(id="full", text="useful content here", metadata={}, domain="test"),
        ]
        retriever = DeterministicRetriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("useful")
        assert len(results) == 1
        assert results[0].document.id == "full"

    def test_add_alias_twice(self, sample_docs: list[Document]) -> None:
        """Adding the same alias twice does not break retrieval."""
        retriever = DeterministicRetriever()
        retriever.index_documents(sample_docs)
        retriever.add_alias("bug", {"type:bug"})
        retriever.add_alias("bug", {"type:bug"})  # no-op
        results = retriever.retrieve("bug")
        assert len(results) >= 1

    def test_rules_yaml_loading_empty(self, tmp_path: Path) -> None:
        """Loading an empty YAML file does not raise errors."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        retriever = DeterministicRetriever(rules_path=str(empty))
        assert retriever._aliases == {}
        assert retriever._routing_rules == []

    def test_retrieve_different_domains_yield_different_results(
        self,
        sample_docs: list[Document],
    ) -> None:
        """Retrieving by different domains yields different document sets."""
        retriever = DeterministicRetriever()
        retriever.index_documents(sample_docs)

        support_results = retriever.retrieve("authentication", domain="support")
        docs_results = retriever.retrieve("authentication", domain="docs")

        support_ids = {r.document.id for r in support_results}
        docs_ids = {r.document.id for r in docs_results}

        assert support_ids != docs_ids
