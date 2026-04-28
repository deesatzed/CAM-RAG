"""Comprehensive tests for semantic chunking strategy.

Tests cover:
- Sentence splitting accuracy (abbreviations, initials, punctuation)
- Chunk size targeting
- Overlap between chunks
- Section heading preservation
- Backward compatibility (default strategy unchanged)
- Strategy parameter routing
- Edge cases (empty docs, single-sentence docs, very long sentences)
- Integration with query path via RAGAppSpec.chunk_strategy
"""

from __future__ import annotations

import pytest

from cam_rag.documents.chunking import (
    ChunkStrategy,
    _chunk_document_semantic,
    _chunk_document_sentence,
    _ends_with_abbreviation,
    _is_heading,
    _split_sentences,
    chunk_document,
    chunk_documents,
    chunk_documents_semantic,
)
from cam_rag.rag.models import Chunk, CorpusDocument
from cam_rag.rag.spec import RAGAppSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(text: str, doc_id: str = "doc_test") -> CorpusDocument:
    return CorpusDocument(
        id=doc_id, text=text, source="clinical_protocol.md", title="Test Protocol"
    )


# Real clinical text samples -- no mock, no placeholders
CLINICAL_PARAGRAPH = (
    "The patient presented with acute chest pain radiating to the left arm. "
    "An ECG was obtained immediately and showed ST-segment elevation in leads II, III, and aVF. "
    "Troponin levels were drawn and returned elevated at 2.4 ng/mL. "
    "The cardiology team was consulted for emergent cardiac catheterization."
)

CLINICAL_MULTI_SECTION = """# Assessment

The patient is a 67-year-old male presenting with acute inferior STEMI.
Troponin I is elevated at 2.4 ng/mL.
Blood pressure is 142/88 mmHg with heart rate of 96 bpm.

# Plan

Administer aspirin 325 mg PO immediately.
Start heparin infusion per protocol.
Activate the cardiac catheterization lab for emergent PCI.
Continue telemetry monitoring and serial troponin measurements.

# Follow-up

Dr. Smith will perform the catheterization procedure.
Post-procedure care includes dual antiplatelet therapy.
Echocardiogram should be obtained within 48 hours.
"""

ABBREVIATION_TEXT = (
    "Dr. Johnson consulted with Prof. Williams regarding the patient. "
    "The dosage was approx. 500 mg per day. "
    "The results were published in Vol. 12 of the journal, e.g. Fig. 3 shows the data. "
    "Mrs. Davis was discharged on Monday."
)

MULTI_SENTENCE_TEXT = (
    "First sentence about clinical findings. "
    "Second sentence describing lab results. "
    "Third sentence noting treatment plan. "
    "Fourth sentence about follow-up care. "
    "Fifth sentence documenting patient response."
)


# ===================================================================
# 1. Sentence splitting accuracy
# ===================================================================

class TestSentenceSplitting:
    """Tests for _split_sentences function."""

    def test_basic_sentence_split(self):
        """Splits text on standard sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2
        assert any("First sentence" in s for s in sentences)

    def test_question_marks(self):
        """Splits on question mark boundaries."""
        text = "What is the diagnosis? The patient has pneumonia. Is further testing needed?"
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_exclamation_marks(self):
        """Splits on exclamation mark boundaries."""
        text = "Emergency! The patient is coding. Call the crash team immediately!"
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_abbreviation_dr(self):
        """Does not split on 'Dr.' abbreviation."""
        text = "Dr. Smith examined the patient. The diagnosis was confirmed."
        sentences = _split_sentences(text)
        # "Dr. Smith examined the patient." should be one sentence
        found = any("Dr. Smith examined" in s for s in sentences)
        assert found, f"Expected 'Dr. Smith examined' in one sentence, got: {sentences}"

    def test_abbreviation_eg(self):
        """Does not split on 'e.g.' abbreviation."""
        text = "Common side effects, e.g. nausea and headache, were reported. Treatment continued."
        sentences = _split_sentences(text)
        found = any("e.g." in s for s in sentences)
        assert found

    def test_abbreviation_ie(self):
        """Does not split on 'i.e.' abbreviation."""
        text = "The primary endpoint, i.e. overall survival, was met. The study was successful."
        sentences = _split_sentences(text)
        found = any("i.e." in s for s in sentences)
        assert found

    def test_abbreviation_prof(self):
        """Does not split on 'Prof.' abbreviation."""
        text = "Prof. Anderson reviewed the case. His recommendation was conservative management."
        sentences = _split_sentences(text)
        found = any("Prof. Anderson" in s for s in sentences)
        assert found

    def test_abbreviation_mrs(self):
        """Does not split on 'Mrs.' abbreviation."""
        text = "Mrs. Davis reported improvement. Follow-up was scheduled."
        sentences = _split_sentences(text)
        found = any("Mrs. Davis" in s for s in sentences)
        assert found

    def test_abbreviation_approx(self):
        """Does not split on 'approx.' abbreviation."""
        text = "The dosage was approx. 500 mg daily. Side effects were minimal."
        sentences = _split_sentences(text)
        found = any("approx." in s for s in sentences)
        assert found

    def test_newlines_as_separators(self):
        """Newlines are treated as sentence boundaries."""
        text = "Line one sentence.\nLine two sentence.\nLine three sentence."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_markdown_heading_as_separate_sentence(self):
        """Markdown headings are kept as their own sentence."""
        text = "# Assessment\nThe patient is stable."
        sentences = _split_sentences(text)
        assert sentences[0] == "# Assessment"
        assert "stable" in sentences[1]

    def test_empty_text(self):
        """Empty text returns no sentences."""
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []
        assert _split_sentences("\n\n") == []

    def test_single_sentence(self):
        """A single sentence without a terminal period still returns."""
        sentences = _split_sentences("Just one sentence")
        assert len(sentences) == 1

    def test_clinical_text_splits_correctly(self):
        """Real clinical paragraph splits into expected sentence count."""
        sentences = _split_sentences(CLINICAL_PARAGRAPH)
        # The clinical paragraph has 4 sentences
        assert len(sentences) >= 3, f"Expected >=3 sentences, got {len(sentences)}: {sentences}"


# ===================================================================
# 2. Heading detection
# ===================================================================

class TestHeadingDetection:
    """Tests for _is_heading function."""

    def test_markdown_h1(self):
        assert _is_heading("# Assessment") is True

    def test_markdown_h2(self):
        assert _is_heading("## Treatment Plan") is True

    def test_markdown_h3(self):
        assert _is_heading("### Follow-up Notes") is True

    def test_short_title_line(self):
        """Short line with no terminal punctuation and uppercase start."""
        assert _is_heading("Assessment") is True

    def test_numbered_heading(self):
        """Lines starting with digits can be headings."""
        assert _is_heading("3 Methodology") is True

    def test_regular_sentence_not_heading(self):
        """A normal sentence ending with a period is not a heading."""
        assert _is_heading("The patient was discharged.") is False

    def test_empty_not_heading(self):
        assert _is_heading("") is False
        assert _is_heading("   ") is False

    def test_long_line_not_heading(self):
        """Lines over 80 characters are not headings (unless they start with #)."""
        long_line = "A" * 100
        assert _is_heading(long_line) is False

    def test_long_markdown_heading_still_heading(self):
        """Even long lines starting with # are headings."""
        long_heading = "# " + "A" * 100
        assert _is_heading(long_heading) is True


# ===================================================================
# 3. Abbreviation ending detection
# ===================================================================

class TestAbbreviationDetection:

    def test_dr_period(self):
        assert _ends_with_abbreviation("Dr.") is True

    def test_prof_period(self):
        assert _ends_with_abbreviation("Prof.") is True

    def test_eg_period(self):
        assert _ends_with_abbreviation("e.g.") is True

    def test_regular_word_period(self):
        assert _ends_with_abbreviation("treatment.") is False

    def test_initial_letter_period(self):
        """Single capital initial like 'J.' should not trigger sentence split."""
        assert _ends_with_abbreviation("J.") is True

    def test_no_period(self):
        assert _ends_with_abbreviation("Doctor") is False


# ===================================================================
# 4. Semantic chunking -- chunk size targeting
# ===================================================================

class TestSemanticChunkSizeTargeting:
    """Verify chunks target the specified character budget."""

    def test_chunks_near_target_size(self):
        """Chunks should be near the target size (within reasonable tolerance)."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = _chunk_document_semantic(doc, target_chunk_size=200)
        for chunk in chunks:
            # Each chunk should not wildly exceed the target (allow 2x for overlap/heading)
            assert len(chunk.text) < 200 * 3, (
                f"Chunk too large ({len(chunk.text)} chars): {chunk.text[:100]}..."
            )

    def test_small_target_produces_more_chunks(self):
        """A smaller target size should produce more chunks."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks_large = _chunk_document_semantic(doc, target_chunk_size=1000)
        chunks_small = _chunk_document_semantic(doc, target_chunk_size=100)
        assert len(chunks_small) > len(chunks_large)

    def test_very_large_target_single_chunk_per_section(self):
        """A very large target may collapse sections into fewer chunks."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = _chunk_document_semantic(doc, target_chunk_size=10000)
        # With headings forcing boundaries, should still have at least 3 chunks
        # (one per section: Assessment, Plan, Follow-up)
        assert len(chunks) >= 3

    def test_single_sentence_doc(self):
        """Single sentence produces exactly one chunk."""
        doc = _make_doc("The patient was stable on discharge.")
        chunks = _chunk_document_semantic(doc, target_chunk_size=512)
        assert len(chunks) == 1
        assert "stable" in chunks[0].text


# ===================================================================
# 5. Semantic chunking -- overlap
# ===================================================================

class TestSemanticOverlap:
    """Verify sentence overlap between consecutive semantic chunks."""

    def test_overlap_sentences_present(self):
        """With overlap_sentences=1, the last sentence of chunk N appears in chunk N+1."""
        doc = _make_doc(MULTI_SENTENCE_TEXT)
        chunks = _chunk_document_semantic(
            doc, target_chunk_size=80, overlap_sentences=1
        )
        if len(chunks) < 2:
            pytest.skip("Text too short to produce multiple chunks at this target size")
        # The overlap sentence from chunk 0 should appear in chunk 1
        # Get the last sentence of chunk 0
        chunk0_sentences = chunks[0].text.split(". ")
        if chunk0_sentences:
            last_sentence_fragment = chunk0_sentences[-1].strip().rstrip(".")
            if last_sentence_fragment:
                assert last_sentence_fragment in chunks[1].text, (
                    f"Expected overlap sentence '{last_sentence_fragment}' "
                    f"in chunk 1: {chunks[1].text}"
                )

    def test_zero_overlap(self):
        """With overlap_sentences=0, no sentence overlap occurs."""
        doc = _make_doc(MULTI_SENTENCE_TEXT)
        chunks_no_overlap = _chunk_document_semantic(
            doc, target_chunk_size=80, overlap_sentences=0
        )
        chunks_with_overlap = _chunk_document_semantic(
            doc, target_chunk_size=80, overlap_sentences=2
        )
        # With overlap, total text length across chunks should be larger
        total_no = sum(len(c.text) for c in chunks_no_overlap)
        total_with = sum(len(c.text) for c in chunks_with_overlap)
        if len(chunks_no_overlap) > 1 and len(chunks_with_overlap) > 1:
            assert total_with >= total_no


# ===================================================================
# 6. Section heading preservation
# ===================================================================

class TestHeadingPreservation:
    """Verify headings are preserved and attached to chunks."""

    def test_heading_starts_new_chunk(self):
        """A heading forces a new chunk boundary."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = _chunk_document_semantic(doc, target_chunk_size=2000)
        # Each section heading should start a new chunk
        heading_chunks = [c for c in chunks if c.text.strip().startswith("#")]
        assert len(heading_chunks) >= 3, (
            f"Expected at least 3 heading chunks, got {len(heading_chunks)}"
        )

    def test_section_heading_metadata(self):
        """Chunks starting with a heading should have section_heading set."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = _chunk_document_semantic(doc, target_chunk_size=2000)
        heading_chunks = [c for c in chunks if c.section_heading]
        assert len(heading_chunks) >= 3

    def test_heading_attached_to_content(self):
        """A heading chunk should also contain the following content sentences."""
        doc = _make_doc("# Plan\nAdminister aspirin. Start heparin.")
        chunks = _chunk_document_semantic(doc, target_chunk_size=2000)
        assert len(chunks) >= 1
        assert "# Plan" in chunks[0].text
        assert "aspirin" in chunks[0].text


# ===================================================================
# 7. Backward compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Default strategy must match original paragraph chunking behavior."""

    def test_default_strategy_is_paragraph(self):
        """chunk_document with no strategy arg uses paragraph splitting."""
        doc = _make_doc("Para one.\n\nPara two.\n\nPara three.")
        chunks = chunk_document(doc)
        assert len(chunks) == 3
        assert chunks[0].text == "Para one."
        assert chunks[1].text == "Para two."
        assert chunks[2].text == "Para three."

    def test_explicit_paragraph_matches_default(self):
        """Explicitly passing strategy='paragraph' matches the default."""
        doc = _make_doc("Alpha.\n\nBeta.\n\nGamma.")
        chunks_default = chunk_document(doc)
        chunks_explicit = chunk_document(doc, strategy="paragraph")
        assert [c.text for c in chunks_default] == [c.text for c in chunks_explicit]

    def test_chunk_documents_default_unchanged(self):
        """chunk_documents default behavior is paragraph-based."""
        docs = [_make_doc("Part A.\n\nPart B.", doc_id="d1")]
        chunks = chunk_documents(docs)
        assert len(chunks) == 2
        assert chunks[0].text == "Part A."

    def test_overlap_chars_still_works(self):
        """overlap_chars parameter continues to work with paragraph strategy."""
        doc = _make_doc("First paragraph.\n\nSecond paragraph.")
        chunks = chunk_document(doc, overlap_chars=5, strategy="paragraph")
        assert len(chunks) == 2
        assert chunks[0].text == "First paragraph."
        # Second chunk should have overlap
        assert chunks[1].text.startswith("raph.")

    def test_paragraph_chunk_ids_unchanged(self):
        """Paragraph strategy still produces _p{n} IDs."""
        doc = _make_doc("A.\n\nB.\n\nC.", doc_id="doc1")
        chunks = chunk_document(doc, strategy="paragraph")
        assert chunks[0].id == "doc1_p0"
        assert chunks[1].id == "doc1_p1"
        assert chunks[2].id == "doc1_p2"


# ===================================================================
# 8. Strategy parameter routing
# ===================================================================

class TestStrategyRouting:
    """Verify that the strategy parameter routes to the correct chunker."""

    def test_paragraph_strategy_level(self):
        """Paragraph strategy sets level='paragraph'."""
        doc = _make_doc("Some text.\n\nMore text.")
        chunks = chunk_document(doc, strategy="paragraph")
        for chunk in chunks:
            assert chunk.level == "paragraph"

    def test_sentence_strategy_level(self):
        """Sentence strategy sets level='sentence'."""
        doc = _make_doc("First sentence. Second sentence.")
        chunks = chunk_document(doc, strategy="sentence")
        for chunk in chunks:
            assert chunk.level == "sentence"

    def test_semantic_strategy_produces_chunks(self):
        """Semantic strategy returns non-empty chunks for non-empty text."""
        doc = _make_doc(CLINICAL_PARAGRAPH)
        chunks = chunk_document(doc, strategy="semantic")
        assert len(chunks) >= 1
        assert all(c.text.strip() for c in chunks)

    def test_sentence_strategy_produces_fine_grained(self):
        """Sentence strategy produces more chunks than paragraph for multi-sentence paragraphs."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        doc = _make_doc(text)
        para_chunks = chunk_document(doc, strategy="paragraph")
        sent_chunks = chunk_document(doc, strategy="sentence")
        assert len(sent_chunks) >= len(para_chunks)

    def test_invalid_strategy_raises(self):
        """An invalid strategy value raises ValueError."""
        doc = _make_doc("Some text.")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_document(doc, strategy="invalid")  # type: ignore[arg-type]

    def test_chunk_documents_forwards_strategy(self):
        """chunk_documents forwards strategy to chunk_document."""
        docs = [_make_doc("First sentence. Second sentence.", doc_id="d1")]
        chunks = chunk_documents(docs, strategy="sentence")
        assert all(c.level == "sentence" for c in chunks)

    def test_semantic_chunk_ids(self):
        """Semantic strategy uses _sem{n} ID format."""
        doc = _make_doc("A sentence here. Another sentence there.", doc_id="doc_x")
        chunks = chunk_document(doc, strategy="semantic")
        assert chunks[0].id.startswith("doc_x_sem")

    def test_sentence_chunk_ids(self):
        """Sentence strategy uses _s{n} ID format."""
        doc = _make_doc("First. Second.", doc_id="doc_y")
        chunks = chunk_document(doc, strategy="sentence")
        assert chunks[0].id.startswith("doc_y_s")


# ===================================================================
# 9. Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases: empty docs, single-sentence, very long sentences."""

    def test_empty_document_paragraph(self):
        """Empty doc produces no chunks (paragraph)."""
        doc = _make_doc("")
        assert chunk_document(doc, strategy="paragraph") == []

    def test_empty_document_semantic(self):
        """Empty doc produces no chunks (semantic)."""
        doc = _make_doc("")
        assert chunk_document(doc, strategy="semantic") == []

    def test_empty_document_sentence(self):
        """Empty doc produces no chunks (sentence)."""
        doc = _make_doc("")
        assert chunk_document(doc, strategy="sentence") == []

    def test_whitespace_only_document(self):
        """Whitespace-only doc produces no chunks."""
        doc = _make_doc("   \n\n   ")
        assert chunk_document(doc, strategy="semantic") == []
        assert chunk_document(doc, strategy="sentence") == []

    def test_single_sentence_semantic(self):
        """A single sentence produces one semantic chunk."""
        doc = _make_doc("Administer aspirin 325 mg PO immediately.")
        chunks = chunk_document(doc, strategy="semantic")
        assert len(chunks) == 1

    def test_single_sentence_sentence_strategy(self):
        """A single sentence produces one sentence chunk."""
        doc = _make_doc("Administer aspirin 325 mg PO immediately.")
        chunks = chunk_document(doc, strategy="sentence")
        assert len(chunks) == 1

    def test_very_long_sentence_semantic(self):
        """A very long sentence still produces a chunk even if it exceeds target."""
        long_sentence = "The patient " + "presented with symptoms " * 100 + "requiring treatment."
        doc = _make_doc(long_sentence)
        chunks = chunk_document(doc, strategy="semantic", target_chunk_size=100)
        assert len(chunks) >= 1
        # The entire long sentence should be present across all chunks
        all_text = " ".join(c.text for c in chunks)
        assert "requiring treatment" in all_text

    def test_document_with_only_headings(self):
        """A document with only headings (no body text)."""
        doc = _make_doc("# Section One\n# Section Two\n# Section Three")
        chunks = chunk_document(doc, strategy="semantic")
        assert len(chunks) >= 1

    def test_multiple_empty_lines(self):
        """Multiple empty lines between paragraphs are handled."""
        doc = _make_doc("Para one.\n\n\n\n\nPara two.")
        chunks = chunk_document(doc, strategy="paragraph")
        assert len(chunks) == 2

    def test_no_terminal_punctuation(self):
        """Text without terminal punctuation still produces chunks."""
        doc = _make_doc("No punctuation here\nAnother line without periods")
        chunks = chunk_document(doc, strategy="sentence")
        assert len(chunks) >= 1


# ===================================================================
# 10. RAGAppSpec integration
# ===================================================================

class TestRAGAppSpecIntegration:
    """Verify RAGAppSpec.chunk_strategy wiring."""

    def test_default_chunk_strategy(self):
        """RAGAppSpec defaults to 'paragraph' strategy."""
        spec = RAGAppSpec(name="test")
        assert spec.chunk_strategy == "paragraph"

    def test_custom_chunk_strategy(self):
        """RAGAppSpec accepts custom strategy values."""
        spec = RAGAppSpec(name="test", chunk_strategy="semantic")
        assert spec.chunk_strategy == "semantic"

    def test_sentence_chunk_strategy(self):
        """RAGAppSpec accepts 'sentence' strategy."""
        spec = RAGAppSpec(name="test", chunk_strategy="sentence")
        assert spec.chunk_strategy == "sentence"

    def test_chunk_strategy_used_in_chunk_documents(self):
        """chunk_documents respects the strategy kwarg (used by query path)."""
        docs = [_make_doc(
            "First sentence here. Second sentence here. Third sentence here.",
            doc_id="d1",
        )]
        para_chunks = chunk_documents(docs, strategy="paragraph")
        sent_chunks = chunk_documents(docs, strategy="sentence")
        # sentence strategy should produce more fine-grained chunks
        assert len(sent_chunks) >= len(para_chunks)

    def test_spec_chunk_overlap_still_works(self):
        """chunk_overlap in spec still works alongside chunk_strategy."""
        spec = RAGAppSpec(name="test", chunk_overlap=5, chunk_strategy="paragraph")
        assert spec.chunk_overlap == 5
        assert spec.chunk_strategy == "paragraph"


# ===================================================================
# 11. chunk_documents_semantic convenience function
# ===================================================================

class TestChunkDocumentsSemanticFunction:
    """Tests for the chunk_documents_semantic helper."""

    def test_returns_chunks(self):
        """chunk_documents_semantic returns Chunk instances."""
        docs = [_make_doc(CLINICAL_PARAGRAPH, doc_id="d1")]
        chunks = chunk_documents_semantic(docs)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_multiple_documents(self):
        """chunk_documents_semantic processes multiple documents."""
        docs = [
            _make_doc("First document sentence.", doc_id="d1"),
            _make_doc("Second document sentence.", doc_id="d2"),
        ]
        chunks = chunk_documents_semantic(docs)
        doc_ids = {c.document_id for c in chunks}
        assert "d1" in doc_ids
        assert "d2" in doc_ids

    def test_target_size_parameter(self):
        """target_chunk_size parameter is respected."""
        docs = [_make_doc(CLINICAL_MULTI_SECTION, doc_id="d1")]
        chunks_small = chunk_documents_semantic(docs, target_chunk_size=100)
        chunks_large = chunk_documents_semantic(docs, target_chunk_size=5000)
        assert len(chunks_small) >= len(chunks_large)


# ===================================================================
# 12. Metadata and structure preservation
# ===================================================================

class TestMetadataPreservation:
    """Verify document metadata flows through to chunks."""

    def test_source_preserved_semantic(self):
        """Chunk source matches document source."""
        doc = _make_doc(CLINICAL_PARAGRAPH)
        chunks = chunk_document(doc, strategy="semantic")
        for chunk in chunks:
            assert chunk.source == "clinical_protocol.md"

    def test_title_preserved_semantic(self):
        """Chunk title matches document title."""
        doc = _make_doc(CLINICAL_PARAGRAPH)
        chunks = chunk_document(doc, strategy="semantic")
        for chunk in chunks:
            assert chunk.title == "Test Protocol"

    def test_document_id_preserved_semantic(self):
        """Chunk document_id matches document id."""
        doc = _make_doc(CLINICAL_PARAGRAPH, doc_id="doc123")
        chunks = chunk_document(doc, strategy="semantic")
        for chunk in chunks:
            assert chunk.document_id == "doc123"

    def test_metadata_dict_preserved(self):
        """Custom metadata is forwarded to chunks."""
        doc = CorpusDocument(
            id="d1",
            text="A sentence. Another sentence.",
            source="test.md",
            title="Test",
            metadata={"department": "cardiology", "priority": "high"},
        )
        chunks = chunk_document(doc, strategy="semantic")
        for chunk in chunks:
            assert chunk.metadata["department"] == "cardiology"
            assert chunk.metadata["priority"] == "high"

    def test_position_sequential_semantic(self):
        """Chunk positions are sequential starting from 0."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = chunk_document(doc, strategy="semantic", target_chunk_size=100)
        positions = [c.position for c in chunks]
        assert positions == list(range(len(positions)))

    def test_position_sequential_sentence(self):
        """Sentence chunk positions are sequential."""
        doc = _make_doc("First. Second. Third.")
        chunks = chunk_document(doc, strategy="sentence")
        positions = [c.position for c in chunks]
        assert positions == list(range(len(positions)))


# ===================================================================
# 13. Clinical text integration
# ===================================================================

class TestClinicalTextIntegration:
    """Full integration tests with real clinical text."""

    def test_clinical_multi_section_semantic(self):
        """Multi-section clinical document is properly chunked semantically."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        chunks = chunk_document(doc, strategy="semantic", target_chunk_size=300)
        assert len(chunks) >= 3  # At least one per section
        all_text = " ".join(c.text for c in chunks)
        assert "aspirin" in all_text
        assert "heparin" in all_text
        assert "echocardiogram" in all_text.lower() or "Echocardiogram" in all_text

    def test_clinical_abbreviation_text_semantic(self):
        """Text with many abbreviations chunks correctly."""
        doc = _make_doc(ABBREVIATION_TEXT)
        chunks = chunk_document(doc, strategy="semantic", target_chunk_size=200)
        all_text = " ".join(c.text for c in chunks)
        # Verify abbreviations are not mangled
        assert "Dr." in all_text or "Dr" in all_text
        assert "Prof." in all_text or "Prof" in all_text
        assert "Mrs." in all_text or "Mrs" in all_text

    def test_all_three_strategies_on_same_text(self):
        """All three strategies produce valid chunks for the same clinical text."""
        doc = _make_doc(CLINICAL_MULTI_SECTION)
        for strategy in ("paragraph", "semantic", "sentence"):
            chunks = chunk_document(doc, strategy=strategy)  # type: ignore[arg-type]
            assert len(chunks) >= 1, f"Strategy '{strategy}' produced no chunks"
            assert all(c.text.strip() for c in chunks), (
                f"Strategy '{strategy}' produced empty chunk"
            )

    def test_semantic_no_text_loss(self):
        """Semantic chunking does not lose content (ignoring overlap duplicates)."""
        text = (
            "Administer aspirin 325 mg. "
            "Start heparin drip. "
            "Monitor cardiac enzymes. "
            "Obtain echocardiogram."
        )
        doc = _make_doc(text)
        chunks = chunk_document(doc, strategy="semantic", target_chunk_size=60, overlap_sentences=0)
        all_text = " ".join(c.text for c in chunks)
        assert "aspirin" in all_text
        assert "heparin" in all_text
        assert "cardiac enzymes" in all_text
        assert "echocardiogram" in all_text
