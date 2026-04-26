"""Thin Ragamuffin app layer.

Ragamuffin owns clinical defaults and document-folder UX. Platform behavior
comes from `cam_rag`.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cam_rag.documents import read_document_folder
from cam_rag.rag import CorpusDocument, RAGAppSpec, RAGPolicy


class QueryBackendMissingError(RuntimeError):
    """Raised when the platform query API has not been wired yet."""


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


def query_documents(question: str, docs_dir: str | Path) -> Any:
    """Ask Ragamuffin's platform-backed query path about a document folder."""

    spec = ragamuffin_spec()
    documents = read_document_folder(docs_dir, spec)
    platform_query = _load_platform_query()
    return platform_query(question=question, documents=documents, spec=spec)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the Ragamuffin app query surface."""

    parser = argparse.ArgumentParser(prog="ragamuffin")
    parser.add_argument("docs_dir", help="Folder containing Ragamuffin source documents")
    parser.add_argument("question", help="Question to ask through the platform query API")
    args = parser.parse_args(argv)

    result = query_documents(args.question, args.docs_dir)
    print(result)
    return 0


def _load_platform_query() -> Callable[..., Any]:
    for module_name in ("cam_rag.query", "cam_rag.simple"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue

        query = getattr(module, "query", None)
        if callable(query):
            return query

    raise QueryBackendMissingError(
        "Ragamuffin requires a platform query API at cam_rag.query.query "
        "or cam_rag.simple.query."
    )
