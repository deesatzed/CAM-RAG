"""Tests for error handling improvements (R6-01)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from cam_rag.documents.folder import read_document_folder
from cam_rag.rag import RAGAppSpec
from cam_rag.retrieval import (
    DenseVectorRetriever,
    RetrievalDocument,
)

# -- folder.py: missing directory raises ValueError --------------------


def test_folder_raises_valueerror_for_missing_directory(
    tmp_path: Path,
):
    missing = tmp_path / "nonexistent"
    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    with pytest.raises(ValueError, match="does not exist"):
        read_document_folder(missing, spec)


# -- folder.py: unreadable file is skipped ----------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 000 does not restrict on Windows",
)
def test_folder_skips_unreadable_file(
    tmp_path: Path, caplog
):
    good = tmp_path / "good.md"
    good.write_text(
        "# Good\nReadable content.", encoding="utf-8"
    )
    bad = tmp_path / "bad.md"
    bad.write_text(
        "# Bad\nSecret content.", encoding="utf-8"
    )
    os.chmod(bad, 0o000)

    spec = RAGAppSpec(
        name="test", supported_extensions=(".md",)
    )
    try:
        with caplog.at_level(
            logging.WARNING,
            logger="cam_rag.documents.folder",
        ):
            docs = read_document_folder(tmp_path, spec)
    finally:
        # Restore permissions so pytest can clean up
        os.chmod(bad, 0o644)

    assert len(docs) == 1
    assert docs[0].source == "good.md"

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    assert any("skipping" in r.message for r in warnings)


# -- folder.py: permission error is handled ----------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 000 does not restrict on Windows",
)
def test_folder_skips_file_with_permission_error(
    tmp_path: Path,
):
    """A file that cannot be read at all is skipped."""
    unreadable = tmp_path / "locked.txt"
    unreadable.write_text(
        "locked content", encoding="utf-8"
    )
    os.chmod(unreadable, 0o000)

    spec = RAGAppSpec(
        name="test", supported_extensions=(".txt",)
    )
    try:
        docs = read_document_folder(tmp_path, spec)
    finally:
        os.chmod(unreadable, 0o644)

    assert docs == []


# -- dense.py: embedding failure uses zero vector ---------------------


class _FailingBackend:
    """Embedding backend that fails on specific texts."""

    dim: int = 64

    def __init__(self, fail_text: str = "FAIL") -> None:
        self._fail_text = fail_text
        mod = __import__(
            "cam_rag.retrieval.dense",
            fromlist=["HashEmbeddingBackend"],
        )
        self._real = mod.HashEmbeddingBackend(dim=self.dim)

    def embed(self, text: str) -> list[float]:
        if self._fail_text in text:
            raise RuntimeError(
                "embedding service unavailable"
            )
        return self._real.embed(text)


def test_dense_retriever_handles_embedding_failure(caplog):
    docs = [
        RetrievalDocument("ok", "triage vital signs"),
        RetrievalDocument(
            "broken", "FAIL embedding here"
        ),
    ]
    with caplog.at_level(
        logging.WARNING,
        logger="cam_rag.retrieval.dense",
    ):
        retriever = DenseVectorRetriever(
            docs, backend=_FailingBackend()
        )

    # The retriever should have been constructed
    assert len(retriever.documents) == 2

    # The broken document should have a zero vector
    assert retriever._vectors[1] == [0.0] * 64

    # A warning should have been logged
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
    ]
    assert any(
        "embedding failed" in r.message for r in warnings
    )


def test_dense_retriever_still_retrieves_after_failure():
    docs = [
        RetrievalDocument("ok", "triage vital signs"),
        RetrievalDocument(
            "broken", "FAIL embedding here"
        ),
    ]
    retriever = DenseVectorRetriever(
        docs, backend=_FailingBackend()
    )

    results = retriever.retrieve(
        "triage vital signs", k=2
    )
    # The good document should still be retrievable
    assert len(results) >= 1
    assert results[0].doc_id == "ok"
