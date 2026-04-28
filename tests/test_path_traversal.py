"""Tests for path traversal guards in folder ingestion and repo mining."""

from __future__ import annotations

import os
from pathlib import Path

from cam_rag.documents.folder import read_document_folder
from cam_rag.methodologies.miner import _iter_candidate_files
from cam_rag.rag.spec import RAGAppSpec


def test_symlink_skipped_in_folder_ingestion(tmp_path: Path):
    """Symlinks inside the document folder must be silently skipped."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "legit.md").write_text("# Legit\nThis is real content.")

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "normal.md").write_text("# Normal\nNormal document content.")
    os.symlink(real_dir / "legit.md", doc_dir / "sneaky_link.md")

    spec = RAGAppSpec(name="test")
    docs = read_document_folder(doc_dir, spec)
    sources = [d.source for d in docs]
    assert "normal.md" in sources
    assert "sneaky_link.md" not in sources


def test_path_escape_skipped_in_folder_ingestion(tmp_path: Path):
    """Files that resolve outside the root must be silently skipped."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret\nSensitive data.")

    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "ok.md").write_text("# OK\nSafe content.")
    os.symlink(outside / "secret.md", doc_dir / "escaped.md")

    spec = RAGAppSpec(name="test")
    docs = read_document_folder(doc_dir, spec)
    sources = [d.source for d in docs]
    assert "ok.md" in sources
    assert "escaped.md" not in sources


def test_symlink_skipped_in_miner(tmp_path: Path):
    """Miner's _iter_candidate_files must skip symlinks."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "real.py").write_text("print('hello')")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.py").write_text("import os; os.system('rm -rf /')")
    os.symlink(outside / "evil.py", repo / "evil_link.py")

    files = list(_iter_candidate_files(repo))
    names = [f.name for f in files]
    assert "real.py" in names
    assert "evil_link.py" not in names
