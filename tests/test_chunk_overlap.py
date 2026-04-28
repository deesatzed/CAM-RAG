"""Tests for P7-03: Chunk overlap support in chunking.py and RAGAppSpec."""

from __future__ import annotations

from cam_rag.documents.chunking import chunk_document, chunk_documents
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec


def _make_doc(text: str, doc_id: str = "doc_test") -> CorpusDocument:
    return CorpusDocument(id=doc_id, text=text, source="test.md", title="Test")


def test_no_overlap_default():
    """With default overlap_chars=0, chunks have no overlap text."""
    doc = _make_doc("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks = chunk_document(doc)
    assert len(chunks) == 3
    assert chunks[0].text == "Paragraph one."
    assert chunks[1].text == "Paragraph two."
    assert chunks[2].text == "Paragraph three."


def test_overlap_prepends_previous_tail():
    """When overlap_chars > 0, the tail of the previous chunk is prepended."""
    doc = _make_doc("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks = chunk_document(doc, overlap_chars=5)
    # First chunk has no predecessor, so it is unchanged
    assert chunks[0].text == "Paragraph one."
    # Second chunk should start with the last 5 chars of chunk 0's text
    assert chunks[1].text.startswith(" one.")
    assert "Paragraph two." in chunks[1].text
    # Third chunk should start with the last 5 chars of chunk 1's text
    assert chunks[2].text.startswith(" two.")
    assert "Paragraph three." in chunks[2].text


def test_overlap_larger_than_chunk():
    """Overlap larger than chunk prepends the entire previous chunk."""
    doc = _make_doc("Hi.\n\nBye.")
    chunks = chunk_document(doc, overlap_chars=1000)
    assert chunks[0].text == "Hi."
    # overlap_chars=1000 but previous chunk is only "Hi." (3 chars)
    # so entire previous text is prepended
    assert chunks[1].text == "Hi.Bye."


def test_overlap_zero_explicit():
    """Explicitly passing overlap_chars=0 behaves the same as default."""
    doc = _make_doc("Alpha.\n\nBeta.")
    chunks_default = chunk_document(doc)
    chunks_explicit = chunk_document(doc, overlap_chars=0)
    assert [c.text for c in chunks_default] == [c.text for c in chunks_explicit]


def test_chunk_documents_passes_overlap():
    """chunk_documents forwards overlap_chars to chunk_document."""
    docs = [
        _make_doc("Part A.\n\nPart B.", doc_id="doc1"),
        _make_doc("Part C.\n\nPart D.", doc_id="doc2"),
    ]
    chunks = chunk_documents(docs, overlap_chars=4)
    # Doc1: chunk0="Part A.", chunk1="t A.Part B."
    assert chunks[0].text == "Part A."
    assert chunks[1].text.startswith("t A.")
    # Doc2: chunk2="Part C." (no overlap -- first chunk of new doc), chunk3 has overlap
    assert chunks[2].text == "Part C."
    assert chunks[3].text.startswith("t C.")


def test_single_paragraph_no_overlap_effect():
    """A single-paragraph document produces one chunk regardless of overlap."""
    doc = _make_doc("Just one paragraph with no double newlines.")
    chunks = chunk_document(doc, overlap_chars=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one paragraph with no double newlines."


def test_rag_app_spec_has_chunk_overlap():
    """RAGAppSpec exposes chunk_overlap with a default of 0."""
    spec = RAGAppSpec(name="test")
    assert spec.chunk_overlap == 0

    spec_with = RAGAppSpec(name="test", chunk_overlap=50)
    assert spec_with.chunk_overlap == 50


def test_overlap_preserves_chunk_ids():
    """Chunk IDs remain stable and sequential even with overlap."""
    doc = _make_doc("A.\n\nB.\n\nC.", doc_id="doc_abc")
    chunks = chunk_document(doc, overlap_chars=2)
    assert chunks[0].id == "doc_abc_p0"
    assert chunks[1].id == "doc_abc_p1"
    assert chunks[2].id == "doc_abc_p2"


def test_overlap_does_not_affect_first_chunk():
    """The first chunk in a document is never modified by overlap."""
    doc = _make_doc("First.\n\nSecond.")
    chunks_no_overlap = chunk_document(doc, overlap_chars=0)
    chunks_with_overlap = chunk_document(doc, overlap_chars=3)
    assert chunks_no_overlap[0].text == chunks_with_overlap[0].text
