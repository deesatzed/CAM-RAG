"""Generic document chunking for the platform query MVP."""

from __future__ import annotations

import re

from cam_rag.rag.models import Chunk, CorpusDocument

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def chunk_document(document: CorpusDocument, *, max_chars: int = 900) -> list[Chunk]:
    """Split a normalized document into paragraph chunks.

    This is the generic baseline chunker. Domain-specific and multi-scale
    chunkers can be plugged in later without changing app code.
    """

    paragraphs = [part.strip() for part in _PARAGRAPH_SPLIT.split(document.text) if part.strip()]
    if not paragraphs and document.text.strip():
        paragraphs = [document.text.strip()]

    chunks: list[Chunk] = []
    position = 0
    for paragraph in paragraphs:
        for text in _split_long_text(paragraph, max_chars=max_chars):
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
            position += 1
    return chunks


def chunk_documents(documents: list[CorpusDocument], *, max_chars: int = 900) -> list[Chunk]:
    """Chunk a list of documents."""

    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_chars=max_chars))
    return chunks


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
