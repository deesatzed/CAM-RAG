"""
Tests for chunk_fractal() — fractal sentence/paragraph chunking.

Validates that the regex-based sentence splitter handles decimals,
abbreviations, statistical notation, and merges short fragments.
No mocks.
"""

import pytest
from cam_rag.retrieval.fractal.core import chunk_fractal


# ============================================================
# Basic functionality
# ============================================================

class TestBasicChunking:
    def test_simple_sentences(self):
        sents, paras, full = chunk_fractal(
            "AI is powerful. Machine learning works well. Neural networks are amazing."
        )
        assert len(sents) >= 1
        assert full == "AI is powerful. Machine learning works well. Neural networks are amazing."

    def test_paragraphs_split_on_double_newline(self):
        _, paras, _ = chunk_fractal("First paragraph.\n\nSecond paragraph.\n\nThird.")
        assert len(paras) == 3

    def test_single_paragraph_fallback(self):
        _, paras, _ = chunk_fractal("All one block no paragraph breaks here.")
        assert len(paras) == 1

    def test_returns_full_text(self):
        text = "Some text here."
        _, _, full = chunk_fractal(text)
        assert full == text

    def test_empty_text(self):
        sents, paras, full = chunk_fractal("")
        assert full == ""

    def test_sentence_count_for_multi_sentence(self):
        text = (
            "Artificial intelligence is transforming every industry. "
            "Machine learning allows systems to learn patterns. "
            "Neural networks power modern breakthroughs. "
            "Large language models represent the frontier."
        )
        sents, _, _ = chunk_fractal(text)
        assert len(sents) == 4


# ============================================================
# Decimal and statistical notation handling
# ============================================================

class TestDecimalHandling:
    def test_decimal_numbers_not_split(self):
        text = "The accuracy was 0.95 on the test set. The model performed well."
        sents, _, _ = chunk_fractal(text)
        # Should NOT split "0." into a separate sentence
        assert all(len(s) > 10 for s in sents)

    def test_percentage_with_decimal(self):
        text = "Sensitivity was 85.3% for the model. Specificity reached 92.1% overall."
        sents, _, _ = chunk_fractal(text)
        assert all(len(s) > 15 for s in sents)

    def test_confidence_interval(self):
        text = "The AUC was 0.79 (95% CI, 0.48-1.31; P = 0.06). This indicates moderate performance."
        sents, _, _ = chunk_fractal(text)
        # CI notation should NOT create garbage fragments
        assert all(len(s) > 20 for s in sents)

    def test_p_value(self):
        text = "Results were significant (P < 0.001). Further analysis confirmed this."
        sents, _, _ = chunk_fractal(text)
        assert all(len(s) > 15 for s in sents)


# ============================================================
# Short fragment merging
# ============================================================

class TestFragmentMerging:
    def test_no_fragments_under_30_chars(self):
        text = (
            "A short bit. And another. "
            "But this is a properly long sentence that should stand alone. "
            "More text here for context."
        )
        sents, _, _ = chunk_fractal(text)
        for s in sents:
            # Allow the first or only sentence to be short
            if len(sents) > 1:
                assert len(s) >= 20 or s == sents[0], f"Fragment too short: {repr(s)}"
