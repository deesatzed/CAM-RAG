"""Thin Ragamuffin app layer.

Ragamuffin owns clinical defaults and document-folder UX. Platform behavior
comes from `cam_rag`.
"""

from __future__ import annotations

from pathlib import Path

from cam_rag.documents import read_document_folder
from cam_rag.rag import CorpusDocument, RAGAppSpec, RAGPolicy


def clinical_tokenizer(text: str) -> list[str]:
    """Small clinical tokenizer placeholder until the full Ragamuffin tokenizer ports."""

    expansions = {
        "ed": "emergency department",
        "icu": "intensive care unit",
        "bbp": "bloodborne pathogen",
    }
    tokens: list[str] = []
    for raw in text.lower().replace("/", " ").split():
        clean = raw.strip(".,:;()[]{}")
        if not clean:
            continue
        tokens.append(clean)
        expanded = expansions.get(clean)
        if expanded:
            tokens.extend(expanded.split())
    return tokens


def ragamuffin_spec() -> RAGAppSpec:
    """Return Ragamuffin's app profile for the shared platform."""

    return RAGAppSpec(
        name="ragamuffin",
        description="Clinical and protocol document-folder RAG",
        supported_extensions=(".md", ".txt", ".json", ".jsonl", ".pdf", ".docx"),
        tokenizer=clinical_tokenizer,
        policy=RAGPolicy(
            enforce_phi=True,
            enforce_pii=True,
            require_citations=True,
            min_confidence=0.55,
            allowed_residency="US",
        ),
        retrieval_top_k=10,
        dense_weight=0.6,
        sparse_weight=0.4,
        reranker_model="ncbi/MedCPT-Cross-Encoder",
        domain_tags=("clinical", "protocols", "document-folder"),
    )


def load_documents(docs_dir: str | Path) -> list[CorpusDocument]:
    """Load a document folder using platform ingestion and Ragamuffin defaults."""

    return read_document_folder(docs_dir, ragamuffin_spec())
