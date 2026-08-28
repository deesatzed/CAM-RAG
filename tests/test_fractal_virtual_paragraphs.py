"""
Tests for virtual paragraph chunking (Phase 3A-1).

When a document has no \\n\\n paragraph breaks but 4+ sentences,
chunk_fractal() should create virtual paragraphs by grouping sentences
into chunks of ~3. This prevents L1 from being degenerate (equal to L2).
"""

import numpy as np
import pytest
from cam_rag.retrieval.fractal.core import chunk_fractal
from cam_rag.retrieval.fractal import FractalRAG, HashEmbedding


# Sentences must be >= 30 chars to avoid short-fragment merging in chunk_fractal
SIX_SENTENCE_TEXT = (
    "Artificial intelligence is transforming healthcare delivery worldwide. "
    "Machine learning algorithms predict patient outcomes accurately. "
    "Natural language processing extracts insights from medical records. "
    "Computer vision systems detect anomalies in radiology scans. "
    "Robotic surgery platforms improve precision during operations. "
    "Ethical frameworks guide responsible deployment of clinical AI."
)

FOUR_SENTENCE_TEXT = (
    "Artificial intelligence is transforming healthcare delivery worldwide. "
    "Machine learning algorithms predict patient outcomes accurately. "
    "Natural language processing extracts insights from medical records. "
    "Computer vision systems detect anomalies in radiology scans."
)


# ============================================================
# chunk_fractal virtual paragraph behavior
# ============================================================

class TestVirtualParagraphChunking:
    def test_single_paragraph_6_sentences_creates_2_virtual(self):
        """6 sentences in 1 paragraph -> 2 virtual paragraphs (3+3)."""
        sents, paras, full = chunk_fractal(SIX_SENTENCE_TEXT)
        assert len(sents) == 6, f"Expected 6 sentences, got {len(sents)}"
        assert len(paras) == 2, f"Expected 2 virtual paragraphs, got {len(paras)}"

    def test_single_paragraph_4_sentences_creates_2_virtual(self):
        """4 sentences -> 2 virtual paragraphs (3+1)."""
        sents, paras, full = chunk_fractal(FOUR_SENTENCE_TEXT)
        assert len(sents) == 4, f"Expected 4 sentences, got {len(sents)}"
        assert len(paras) == 2, f"Expected 2 virtual paragraphs, got {len(paras)}"

    def test_single_paragraph_3_sentences_no_virtual(self):
        """3 sentences (below threshold) -> stays as 1 paragraph."""
        text = (
            "Artificial intelligence is transforming healthcare delivery worldwide. "
            "Machine learning algorithms predict patient outcomes accurately. "
            "Natural language processing extracts insights from medical records."
        )
        sents, paras, full = chunk_fractal(text)
        assert len(paras) == 1

    def test_single_paragraph_1_sentence_no_virtual(self):
        """1 sentence -> stays as 1 paragraph."""
        text = "Just one long sentence that has no paragraph breaks at all and is quite long."
        sents, paras, full = chunk_fractal(text)
        assert len(paras) == 1

    def test_multi_paragraph_doc_unchanged(self):
        """Multi-paragraph docs should NOT get virtual paragraphs."""
        text = (
            "First paragraph has content about artificial intelligence methods.\n\n"
            "Second paragraph covers machine learning applications in detail.\n\n"
            "Third paragraph discusses natural language processing techniques."
        )
        sents, paras, full = chunk_fractal(text)
        assert len(paras) == 3

    def test_empty_text_safe(self):
        """Empty text should not crash."""
        sents, paras, full = chunk_fractal("")
        assert full == ""

    def test_virtual_paragraph_text_is_sentence_concatenation(self):
        """Virtual paragraphs should be the concatenation of source sentences."""
        sents, paras, full = chunk_fractal(FOUR_SENTENCE_TEXT)
        assert len(paras) == 2
        # First virtual paragraph should contain the first 3 sentences
        for sent in sents[:3]:
            assert sent in paras[0], f"Expected '{sent}' in first virtual paragraph"
        # Last sentence should be in the second virtual paragraph
        assert sents[3] in paras[1], f"Expected last sentence in second virtual paragraph"


# ============================================================
# Integration with FractalRAG indexing
# ============================================================

class TestVirtualParagraphIndexing:
    def test_single_paragraph_doc_has_multiple_l1_entries(self):
        """A single-paragraph doc with 6 sentences should have 2+ L1 entries."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", SIX_SENTENCE_TEXT)
        stats = rag.stats()
        assert stats["entries_level_1"] >= 2, (
            f"Expected 2+ L1 entries for virtual paragraphs, got {stats['entries_level_1']}"
        )

    def test_derivatives_noncollapsed_for_virtual_paragraphs(self):
        """With virtual paragraphs, d2 should not collapse to near-zero."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        rag.add_document("d1", SIX_SENTENCE_TEXT)

        # Check that at least some d2 derivatives have non-trivial magnitude
        d2_norms = []
        for sid, derivs in rag.derivatives.items():
            d2_norms.append(float(np.linalg.norm(derivs["d2"])))

        assert len(d2_norms) > 0
        mean_d2_norm = np.mean(d2_norms)
        assert mean_d2_norm > 0.01, (
            f"d2 norms too small ({mean_d2_norm:.4f}), "
            "suggesting derivatives collapsed despite virtual paragraphs"
        )

    def test_l1_count_increases_for_single_paragraph_corpus(self):
        """Compare L1 count: multi docs, some single-paragraph, all get virtual paragraphs."""
        rag = FractalRAG(backend=HashEmbedding(dim=64))
        # Two single-paragraph docs, each with 4+ sentences
        rag.add_document("d1", FOUR_SENTENCE_TEXT)
        rag.add_document("d2", SIX_SENTENCE_TEXT)
        stats = rag.stats()
        # d1: 4 sentences -> 2 virtual paragraphs, d2: 6 sentences -> 2 virtual paragraphs
        assert stats["entries_level_1"] >= 4, (
            f"Expected 4+ L1 entries across 2 docs, got {stats['entries_level_1']}"
        )
        assert stats["entries_level_2"] == 2
