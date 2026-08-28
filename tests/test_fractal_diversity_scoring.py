"""
Tests for cross-document diversity scoring (_apply_domain_diversity).

Direction 3: Domain-based MMR reranking for synthesis queries.
All tests use real HashEmbedding. No mocks.
"""

import numpy as np
import pytest

from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding


def _make_rag(dim=64):
    return FractalRAG(backend=HashEmbedding(dim=dim))


def _build_multi_domain_rag(dim=64):
    """Build a RAG with documents across 4 domains."""
    rag = _make_rag(dim)
    docs = [
        ("d1", "AI detects tumors in radiology scans using deep learning models and convolutional neural networks.",
         {"domain": "radiology"}, "Radiology AI"),
        ("d2", "CT imaging reveals lung nodules detected by computer vision algorithms trained on medical datasets.",
         {"domain": "radiology"}, "CT Imaging"),
        ("d3", "Drug metabolism is affected by genetic polymorphisms in cytochrome P450 enzyme pathways.",
         {"domain": "pharma"}, "Pharma Genetics"),
        ("d4", "Pharmacovigilance monitors adverse drug reactions using signal detection methods and databases.",
         {"domain": "pharma"}, "Drug Safety"),
        ("d5", "Clinical decision support systems improve patient outcomes through evidence-based triage protocols.",
         {"domain": "clinical"}, "Clinical DSS"),
        ("d6", "Emergency department triage uses machine learning to prioritize patients by severity score.",
         {"domain": "clinical"}, "ED Triage"),
        ("d7", "Natural language processing extracts clinical entities from electronic health records.",
         {"domain": "nlp"}, "Clinical NLP"),
        ("d8", "Text mining identifies drug-drug interactions from medical literature corpora.",
         {"domain": "nlp"}, "Drug Text Mining"),
    ]
    for doc_id, text, meta, title in docs:
        rag.add_document(doc_id, text, metadata=meta, title=title)
    return rag


class TestApplyDomainDiversity:
    def test_returns_k_results(self):
        """_apply_domain_diversity should return exactly k results."""
        rag = _build_multi_domain_rag()
        scored = [("d1", 0.9), ("d2", 0.85), ("d3", 0.8), ("d4", 0.75),
                  ("d5", 0.7), ("d6", 0.65), ("d7", 0.6), ("d8", 0.55)]
        result = rag._apply_domain_diversity(scored, k=5, lambda_diversity=0.3)
        assert len(result) == 5

    def test_zero_diversity_preserves_order(self):
        """With diversity_boost=0, order should be purely by relevance."""
        rag = _build_multi_domain_rag()
        scored = [("d1", 0.9), ("d2", 0.85), ("d3", 0.8), ("d4", 0.75),
                  ("d5", 0.7), ("d6", 0.65)]
        result = rag._apply_domain_diversity(scored, k=5, lambda_diversity=0.0)
        result_ids = [doc_id for doc_id, _ in result]
        assert result_ids == ["d1", "d2", "d3", "d4", "d5"]

    def test_diversity_promotes_different_domains(self):
        """With diversity_boost > 0, multiple domains should appear in top-k."""
        rag = _build_multi_domain_rag()
        # All radiology docs ranked highest by relevance
        scored = [("d1", 0.9), ("d2", 0.88), ("d3", 0.5), ("d4", 0.48),
                  ("d5", 0.45), ("d6", 0.43), ("d7", 0.4), ("d8", 0.38)]
        result = rag._apply_domain_diversity(scored, k=4, lambda_diversity=0.4)
        result_ids = [doc_id for doc_id, _ in result]

        # Should have docs from different domains, not just radiology
        domains = set()
        for doc_id, _ in result:
            domain = rag.doc_metadata.get(doc_id, {}).get("domain")
            if domain:
                domains.add(domain)
        assert len(domains) >= 3, f"Expected >= 3 domains, got {domains}"

    def test_domain_novelty_is_binary(self):
        """First doc from a domain gets novelty=1.0, subsequent get 0.0."""
        rag = _build_multi_domain_rag()
        # Two radiology docs with similar scores
        scored = [("d1", 0.9), ("d3", 0.85), ("d2", 0.84), ("d5", 0.8)]
        result = rag._apply_domain_diversity(scored, k=4, lambda_diversity=0.5)
        result_ids = [doc_id for doc_id, _ in result]

        # d1 (radiology) should come first (highest relevance + novelty)
        assert result_ids[0] == "d1"
        # d3 (pharma, new domain) should come before d2 (radiology, seen domain)
        d3_pos = result_ids.index("d3")
        d2_pos = result_ids.index("d2")
        assert d3_pos < d2_pos, f"d3 (new domain) should rank above d2 (seen domain)"

    def test_docs_without_metadata_get_zero_novelty(self):
        """Documents without metadata get novelty=0.0 (safe fallback)."""
        rag = _make_rag()
        rag.add_document("d1", "Some text about AI.", metadata={"domain": "ai"})
        rag.add_document("d2", "More text about AI.")  # No metadata
        rag.add_document("d3", "Clinical text.", metadata={"domain": "clinical"})

        scored = [("d1", 0.9), ("d2", 0.85), ("d3", 0.5)]
        result = rag._apply_domain_diversity(scored, k=3, lambda_diversity=0.3)
        # Should not crash, d2 gets novelty=0.0
        assert len(result) == 3

    def test_all_same_domain_no_reranking_effect(self):
        """When all docs are same domain, diversity can't rerank differently
        after the first pick (all get novelty=0.0 after first)."""
        rag = _make_rag()
        for i in range(5):
            rag.add_document(f"d{i}", f"Text about radiology topic {i}.",
                             metadata={"domain": "radiology"})

        scored = [(f"d{i}", 0.9 - i * 0.05) for i in range(5)]
        result = rag._apply_domain_diversity(scored, k=4, lambda_diversity=0.3)
        result_ids = [doc_id for doc_id, _ in result]
        # After first pick, all remaining have novelty=0 so relevance order holds
        assert result_ids == ["d0", "d1", "d2", "d3"]

    def test_fewer_candidates_than_k(self):
        """When fewer candidates than k, return all of them."""
        rag = _make_rag()
        rag.add_document("d1", "Text.", metadata={"domain": "ai"})
        rag.add_document("d2", "More.", metadata={"domain": "bio"})

        scored = [("d1", 0.9), ("d2", 0.8)]
        result = rag._apply_domain_diversity(scored, k=5, lambda_diversity=0.3)
        assert len(result) == 2

    def test_retrieve_reranked_with_diversity_boost(self):
        """End-to-end: retrieve_reranked with diversity_boost > 0 works."""
        rag = _build_multi_domain_rag()
        results, qtype = rag.retrieve_reranked(
            "Compare AI applications across radiology and drug discovery",
            k=5,
            diversity_boost=0.3,
        )
        assert len(results[2]) > 0

        # With diversity, we should see multiple domains in results
        domains = set()
        for entry, score in results[2]:
            domain = rag.doc_metadata.get(entry.id, {}).get("domain")
            if domain:
                domains.add(domain)
        assert len(domains) >= 2, f"Expected >= 2 domains, got {domains}"

    def test_retrieve_reranked_no_diversity_default(self):
        """Default diversity_boost=0.0 should not change behavior."""
        rag = _build_multi_domain_rag()
        # Without diversity
        r1, _ = rag.retrieve_reranked("radiology imaging", k=5, diversity_boost=0.0)
        # Also without specifying it (default)
        r2, _ = rag.retrieve_reranked("radiology imaging", k=5)

        scores1 = {e.id: s for e, s in r1[2]}
        scores2 = {e.id: s for e, s in r2[2]}
        assert scores1 == scores2

    def test_high_diversity_maximizes_domain_coverage(self):
        """With very high diversity (0.9), should get one doc per domain."""
        rag = _build_multi_domain_rag()
        # 4 domains, so top-4 should be 4 different domains
        scored = [("d1", 0.9), ("d2", 0.88), ("d3", 0.5), ("d4", 0.48),
                  ("d5", 0.45), ("d6", 0.43), ("d7", 0.4), ("d8", 0.38)]
        result = rag._apply_domain_diversity(scored, k=4, lambda_diversity=0.9)

        domains = set()
        for doc_id, _ in result:
            domain = rag.doc_metadata.get(doc_id, {}).get("domain")
            if domain:
                domains.add(domain)
        assert len(domains) == 4, f"Expected 4 unique domains, got {domains}"
