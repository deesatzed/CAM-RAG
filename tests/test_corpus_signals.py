"""Tests for CorpusSignals dataclass and compute_corpus_signals function."""

from __future__ import annotations

import statistics

import pytest

from cam_rag.retrieval.corpus_signals import CorpusSignals, compute_corpus_signals


def _simple_tokenizer(text: str) -> list[str]:
    """Simple whitespace tokenizer for tests."""
    return text.lower().split()


class TestCorpusSignalsDataclass:
    def test_frozen(self) -> None:
        signals = CorpusSignals(
            avg_doc_length=50.0,
            doc_length_std=10.0,
            corpus_size=5,
            vocab_overlap_ratio=0.3,
            unique_term_ratio=0.2,
            short_doc_fraction=0.6,
        )
        with pytest.raises(AttributeError):
            signals.avg_doc_length = 99.0  # type: ignore[misc]

    def test_slots(self) -> None:
        assert hasattr(CorpusSignals, "__slots__")

    def test_fields(self) -> None:
        expected_fields = {
            "avg_doc_length",
            "doc_length_std",
            "corpus_size",
            "vocab_overlap_ratio",
            "unique_term_ratio",
            "short_doc_fraction",
        }
        actual_fields = {f.name for f in CorpusSignals.__dataclass_fields__.values()}
        assert actual_fields == expected_fields


class TestComputeCorpusSignals:
    def test_empty_corpus(self) -> None:
        signals = compute_corpus_signals([], _simple_tokenizer)
        assert signals.avg_doc_length == 0.0
        assert signals.doc_length_std == 0.0
        assert signals.corpus_size == 0
        assert signals.vocab_overlap_ratio == 0.0
        assert signals.unique_term_ratio == 0.0
        assert signals.short_doc_fraction == 0.0

    def test_single_document(self) -> None:
        docs = ["the quick brown fox jumps over the lazy dog"]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.corpus_size == 1
        assert signals.avg_doc_length == 9.0
        assert signals.doc_length_std == 0.0
        # Single doc: all terms are in 100% of docs
        assert signals.vocab_overlap_ratio == 1.0

    def test_short_documents_high_fraction(self) -> None:
        # Each doc has far fewer than 200 tokens
        docs = [
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
            "lambda mu nu xi omicron",
            "pi rho sigma tau upsilon",
            "phi chi psi omega zed",
        ]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.short_doc_fraction == pytest.approx(1.0)

    def test_long_documents_low_fraction(self) -> None:
        # Each doc has 300+ tokens (far above the 200-token threshold)
        long_text = " ".join(f"word{i}" for i in range(350))
        docs = [long_text for _ in range(5)]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.short_doc_fraction == pytest.approx(0.0)

    def test_mixed_documents(self) -> None:
        short = "this is a short document with only a few words"
        long_text = " ".join(f"term{i}" for i in range(250))
        # 2 short, 3 long
        docs = [short, short, long_text, long_text, long_text]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert 0.0 < signals.short_doc_fraction < 1.0
        assert signals.short_doc_fraction == pytest.approx(2 / 5)

    def test_vocab_overlap_high_for_repetitive_corpus(self) -> None:
        # All 10 docs share the exact same words
        repeated = "the cat sat on the mat"
        docs = [repeated for _ in range(10)]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        # Every term appears in 100% of docs (>10%), so overlap ratio = 1.0
        assert signals.vocab_overlap_ratio == pytest.approx(1.0)

    def test_vocab_overlap_low_for_diverse_corpus(self) -> None:
        # 10 docs, each with completely unique words (no overlap at all)
        docs = [
            " ".join(f"unique{i}word{j}" for j in range(20))
            for i in range(10)
        ]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        # Each term appears in exactly 1 doc; threshold = 0.1 * 10 = 1.
        # df > 1 is required, so no term qualifies.
        assert signals.vocab_overlap_ratio == pytest.approx(0.0)

    def test_corpus_size_correct(self) -> None:
        docs = ["doc one", "doc two", "doc three", "doc four"]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.corpus_size == 4

    def test_avg_doc_length(self) -> None:
        docs = [
            "one two three",         # 3 tokens
            "one two three four",    # 4 tokens
            "one two three four five",  # 5 tokens
        ]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.avg_doc_length == pytest.approx(4.0)
        # Verify std as well
        expected_std = statistics.stdev([3, 4, 5])
        assert signals.doc_length_std == pytest.approx(expected_std)

    def test_unique_term_ratio_high_diversity(self) -> None:
        # Every token is unique: ratio = vocab_size / total_tokens = 1.0
        docs = ["alpha beta gamma", "delta epsilon zeta"]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        # 6 unique terms, 6 total tokens
        assert signals.unique_term_ratio == pytest.approx(1.0)

    def test_unique_term_ratio_low_diversity(self) -> None:
        # Many repeats of the same token
        docs = ["the the the the the", "the the the the the"]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        # 1 unique term, 10 total tokens => ratio = 0.1
        assert signals.unique_term_ratio == pytest.approx(0.1)

    def test_nfcorpus_like_scenario(self) -> None:
        """Short biomedical-style documents should produce high
        short_doc_fraction and moderate vocab_overlap."""
        docs = [
            "aspirin reduces platelet aggregation and prevents thrombosis",
            "metformin lowers blood glucose levels in type 2 diabetes",
            "ibuprofen inhibits cyclooxygenase and reduces inflammation",
            "amoxicillin is a broad spectrum beta lactam antibiotic",
            "lisinopril is an ace inhibitor used for hypertension treatment",
            "atorvastatin reduces cholesterol synthesis via hmg coa reductase",
            "omeprazole inhibits the gastric proton pump and reduces acid",
            "metformin improves insulin sensitivity in diabetic patients",
            "aspirin is used for secondary prevention of cardiovascular events",
            "ibuprofen is a nonsteroidal anti inflammatory drug nsaid",
        ]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        # All docs are very short (<200 tokens)
        assert signals.short_doc_fraction == pytest.approx(1.0)
        # Some terms like "reduces", "inhibits" appear across multiple docs
        assert signals.vocab_overlap_ratio > 0.0
        assert signals.corpus_size == 10

    def test_scifact_like_scenario(self) -> None:
        """Longer scientific docs should produce low short_doc_fraction."""
        base_abstract = (
            "We conducted a randomized controlled trial to evaluate the "
            "efficacy and safety of the novel compound in a cohort of "
            "patients diagnosed with the condition. The primary endpoint "
            "was defined as a statistically significant reduction in the "
            "biomarker levels measured at baseline and follow up visits. "
            "Secondary endpoints included quality of life scores and "
            "adverse event rates. Results demonstrated a meaningful "
            "improvement compared to placebo with an acceptable safety "
            "profile. Subgroup analyses revealed consistent benefits "
            "across age gender and disease severity strata. These "
            "findings support further investigation in phase three "
            "clinical trials. "
        )
        # Repeat the base text ~3x to push each doc well above 200 tokens
        docs = [
            (base_abstract * 3) + f" study {i} conclusion {i}"
            for i in range(8)
        ]
        signals = compute_corpus_signals(docs, _simple_tokenizer)
        assert signals.short_doc_fraction == pytest.approx(0.0)
        assert signals.corpus_size == 8
        # High overlap since all docs share the same base vocabulary
        assert signals.vocab_overlap_ratio > 0.5
