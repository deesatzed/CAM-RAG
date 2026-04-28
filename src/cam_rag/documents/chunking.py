"""Generic document chunking for the platform query MVP.

Supports multiple chunking strategies:

- ``paragraph`` (default): split on double-newline boundaries, the original
  baseline chunker.
- ``semantic``: group adjacent sentences into chunks that target a character
  budget while preserving section headings and providing sentence overlap for
  context continuity.
- ``sentence``: split into individual sentences (fine-grained retrieval).
"""

from __future__ import annotations

import re
from typing import Literal

from cam_rag.rag.models import Chunk, CorpusDocument

ChunkStrategy = Literal["paragraph", "semantic", "sentence"]

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger a sentence break.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs",
    "e.g", "i.e", "etc", "al", "approx", "dept", "est",
    "inc", "ltd", "no", "vol", "fig", "eq", "ref",
})

# Regex that splits on sentence-ending punctuation followed by whitespace.
# Uses a negative look-behind for single capital letters (e.g. "A. Smith")
# to avoid splitting on initials.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'\(\[])"
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences respecting common abbreviations.

    The splitter handles:
    - Standard sentence endings (``. `` / ``? `` / ``! ``)
    - Abbreviations like ``Dr.``, ``e.g.``, ``i.e.`` (no false split)
    - Newlines as sentence separators
    - Lines starting with ``#`` (Markdown headings) kept as separate units
    """
    if not text or not text.strip():
        return []

    # First split on newlines to respect line structure
    lines = text.split("\n")
    sentences: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Markdown headings are always their own sentence
        if stripped.startswith("#"):
            sentences.append(stripped)
            continue

        # Split the line on sentence boundaries
        raw_parts = _SENTENCE_BOUNDARY.split(stripped)
        # Rejoin incorrectly split abbreviations
        merged: list[str] = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            if merged and _ends_with_abbreviation(merged[-1]):
                merged[-1] = merged[-1] + " " + part
            else:
                merged.append(part)
        sentences.extend(merged)

    return sentences


def _ends_with_abbreviation(text: str) -> bool:
    """Return True if *text* ends with a known abbreviation followed by a period."""
    # Check the last word before the period
    if not text.endswith("."):
        return False
    # Get the last token
    tokens = re.findall(r"[A-Za-z][A-Za-z.]*", text)
    if not tokens:
        return False
    last_token = tokens[-1].rstrip(".")
    if last_token.lower() in _ABBREVIATIONS:
        return True
    # Single capital letter followed by period (initial): "A." "J."
    if len(last_token) == 1 and last_token.isupper():
        return True
    return False


def _is_heading(line: str) -> bool:
    """Return True if *line* looks like a section heading.

    A heading is either a Markdown heading (starts with ``#``) or a short
    line (fewer than 80 characters, no ending punctuation) that looks like
    a title.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    # Short line with no terminal punctuation => likely a heading
    if len(stripped) < 80 and not stripped[-1] in ".!?:;,":
        # Must also not be a single sentence fragment that's just short
        # Heuristic: headings don't usually start with lowercase unless after #
        if stripped[0].isupper() or stripped[0].isdigit():
            return True
    return False


# ---------------------------------------------------------------------------
# Strategy: paragraph (original)
# ---------------------------------------------------------------------------

def _chunk_document_paragraph(
    document: CorpusDocument, *, max_chars: int = 900, overlap_chars: int = 0
) -> list[Chunk]:
    """Split a normalized document into paragraph chunks (original strategy)."""

    paragraphs = [
        part.strip()
        for part in _PARAGRAPH_SPLIT.split(document.text)
        if part.strip()
    ]
    if not paragraphs and document.text.strip():
        paragraphs = [document.text.strip()]

    chunks: list[Chunk] = []
    position = 0
    previous_text: str = ""
    for paragraph in paragraphs:
        for text in _split_long_text(paragraph, max_chars=max_chars):
            if overlap_chars > 0 and previous_text:
                overlap = previous_text[-overlap_chars:]
                text = overlap + text
            chunks.append(
                Chunk(
                    id=f"{document.id}_p{position}",
                    document_id=document.id,
                    text=text,
                    level="paragraph",
                    source=document.source,
                    title=document.title,
                    position=position,
                    metadata=dict(document.metadata),
                )
            )
            previous_text = text
            position += 1
    return chunks


# ---------------------------------------------------------------------------
# Strategy: sentence
# ---------------------------------------------------------------------------

def _chunk_document_sentence(
    document: CorpusDocument,
) -> list[Chunk]:
    """Split a document into individual sentence chunks."""
    sentences = _split_sentences(document.text)
    chunks: list[Chunk] = []
    for position, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        chunks.append(
            Chunk(
                id=f"{document.id}_s{position}",
                document_id=document.id,
                text=sentence,
                level="sentence",
                source=document.source,
                title=document.title,
                position=position,
                metadata=dict(document.metadata),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy: semantic
# ---------------------------------------------------------------------------

def _chunk_document_semantic(
    document: CorpusDocument,
    *,
    target_chunk_size: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Group adjacent sentences into semantically coherent chunks.

    **Algorithm:**

    1. Split the document text into sentences.
    2. Walk through sentences, accumulating them into a chunk until adding
       another sentence would exceed *target_chunk_size* characters.
    3. When a section heading is encountered, start a new chunk and attach the
       heading to the beginning of that chunk.
    4. Between consecutive chunks that are NOT separated by a heading, repeat
       the last *overlap_sentences* sentences of the previous chunk at the
       beginning of the next chunk for context continuity.  Headings always
       start a clean section boundary (no overlap carried across headings).
    """
    sentences = _split_sentences(document.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    position = 0

    # Track current chunk being built
    current_sentences: list[str] = []
    current_len = 0

    def _flush() -> list[str]:
        """Emit the current accumulator as a chunk and return overlap sentences."""
        nonlocal position, current_sentences, current_len

        if not current_sentences:
            return []

        text = " ".join(current_sentences)
        # Determine section heading for metadata
        heading = ""
        if current_sentences and _is_heading(current_sentences[0]):
            heading = current_sentences[0]

        chunks.append(
            Chunk(
                id=f"{document.id}_sem{position}",
                document_id=document.id,
                text=text,
                level="paragraph",  # Use paragraph level for compatibility
                source=document.source,
                title=document.title,
                position=position,
                section_heading=heading,
                metadata=dict(document.metadata),
            )
        )
        position += 1

        # Compute overlap: last N sentences of the current chunk
        n = min(overlap_sentences, len(current_sentences))
        overlap = list(current_sentences[-n:]) if n > 0 else []

        current_sentences = []
        current_len = 0
        return overlap

    overlap_carry: list[str] = []

    for sentence in sentences:
        is_head = _is_heading(sentence)

        # A heading forces a new chunk boundary
        if is_head and current_sentences:
            _flush()
            # Discard overlap across heading boundaries -- headings start
            # a clean section and should not carry context from the
            # previous section.
            overlap_carry = []

        # If this is the first sentence in a new chunk and we have overlap
        # to prepend, do so -- but only if the current sentence is NOT a
        # heading (headings start clean).
        if not current_sentences and overlap_carry and not is_head:
            for ov in overlap_carry:
                current_sentences.append(ov)
                current_len += len(ov) + 1
            overlap_carry = []

        # Check if adding this sentence would exceed the target
        added_len = len(sentence) + (1 if current_sentences else 0)
        if current_sentences and current_len + added_len > target_chunk_size and not is_head:
            # Flush the current chunk and carry overlap forward
            overlap_carry = _flush()
            # Prepend overlap to the new chunk
            if overlap_carry:
                for ov in overlap_carry:
                    current_sentences.append(ov)
                    current_len += len(ov) + 1
                overlap_carry = []

        current_sentences.append(sentence)
        current_len += len(sentence) + (1 if len(current_sentences) > 1 else 0)

    # Flush remaining
    if current_sentences:
        _flush()

    return chunks


def chunk_documents_semantic(
    documents: list[CorpusDocument],
    *,
    target_chunk_size: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Chunk a list of documents using the semantic strategy."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            _chunk_document_semantic(
                document,
                target_chunk_size=target_chunk_size,
                overlap_sentences=overlap_sentences,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    document: CorpusDocument,
    *,
    max_chars: int = 900,
    overlap_chars: int = 0,
    strategy: ChunkStrategy = "paragraph",
    target_chunk_size: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split a normalized document into chunks.

    Parameters
    ----------
    document:
        The document to chunk.
    max_chars:
        Maximum characters per chunk (paragraph strategy only).
    overlap_chars:
        Character overlap between consecutive paragraph chunks.
    strategy:
        Chunking strategy: ``"paragraph"`` (default), ``"semantic"``, or
        ``"sentence"``.
    target_chunk_size:
        Target character budget per chunk (semantic strategy only).
    overlap_sentences:
        Number of overlapping sentences between semantic chunks.

    Returns
    -------
    list[Chunk]
    """
    if strategy == "paragraph":
        return _chunk_document_paragraph(
            document, max_chars=max_chars, overlap_chars=overlap_chars,
        )
    if strategy == "semantic":
        return _chunk_document_semantic(
            document,
            target_chunk_size=target_chunk_size,
            overlap_sentences=overlap_sentences,
        )
    if strategy == "sentence":
        return _chunk_document_sentence(document)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_documents(
    documents: list[CorpusDocument],
    *,
    max_chars: int = 900,
    overlap_chars: int = 0,
    strategy: ChunkStrategy = "paragraph",
    target_chunk_size: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Chunk a list of documents.

    Delegates to :func:`chunk_document` per document.  All keyword arguments
    are forwarded.  After chunking, each chunk is scored for complexity
    (stored in ``chunk.metadata["complexity"]`` and
    ``chunk.metadata["complexity_score"]``).
    """
    from cam_rag.documents.complexity import score_chunks

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                strategy=strategy,
                target_chunk_size=target_chunk_size,
                overlap_sentences=overlap_sentences,
            )
        )
    score_chunks(chunks)
    return chunks


# ---------------------------------------------------------------------------
# Internal helpers (paragraph strategy)
# ---------------------------------------------------------------------------

def _split_long_text(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]
