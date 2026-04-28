"""Folder-based document ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

from cam_rag.documents.parsers import (
    extract_title,
    parse_docx_text,
    parse_pdf_text,
)
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".md", ".txt"}
JSON_EXTENSIONS = {".json", ".jsonl"}
PARSER_EXTENSIONS = {".pdf", ".docx"}


def read_document_folder(
    path: str | Path, spec: RAGAppSpec
) -> list[CorpusDocument]:
    """Read supported files from a folder into platform documents.

    Supports text/markdown, common JSON export shapes, and optional
    PDF/DOCX parser backends.

    Raises ``ValueError`` if *path* does not exist.
    Raises ``NotADirectoryError`` if *path* is not a directory.
    """

    root = Path(path)
    if not root.exists():
        raise ValueError(
            f"documents directory does not exist: {root}"
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"document folder is not a directory: {root}"
        )

    supported = {
        ext.lower() for ext in spec.supported_extensions
    }
    documents: list[CorpusDocument] = []
    root_resolved = root.resolve()
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        # Reject symlinks and paths that escape the document root
        if file_path.is_symlink() or not str(
            file_path.resolve()
        ).startswith(str(root_resolved) + "/"):
            continue
        ext = file_path.suffix.lower()
        if ext not in supported:
            continue
        if ext in TEXT_EXTENSIONS:
            doc = _safe_read_text(root, file_path)
            if doc is not None and doc.text.strip():
                documents.append(doc)
        elif ext in JSON_EXTENSIONS:
            documents.extend(
                _read_json_documents(root, file_path)
            )
        elif ext in PARSER_EXTENSIONS:
            doc = _safe_read_parsed(root, file_path)
            if doc is not None and doc.text.strip():
                documents.append(doc)
    return documents


# -- safe wrappers that skip unreadable files --------------------------


def _safe_read_text(
    root: Path, file_path: Path
) -> CorpusDocument | None:
    """Read a text file, returning ``None`` on IO or decode errors."""
    try:
        return _read_text_document(root, file_path)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "skipping unreadable text file %s: %s",
            file_path,
            exc,
        )
        return None


def _safe_read_parsed(
    root: Path, file_path: Path
) -> CorpusDocument | None:
    """Read a parsed (PDF/DOCX) file, returning ``None`` on error."""
    try:
        return _read_parsed_document(root, file_path)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "skipping unreadable parsed file %s: %s",
            file_path,
            exc,
        )
        return None


# -- core readers (unchanged logic) ------------------------------------


def _read_text_document(
    root: Path, file_path: Path
) -> CorpusDocument:
    text = file_path.read_text(encoding="utf-8")
    rel = str(file_path.relative_to(root))
    return CorpusDocument(
        id=_stable_id(rel, text),
        text=text,
        source=rel,
        title=_extract_markdown_title(text) or file_path.stem,
        format=file_path.suffix.lower().lstrip("."),
        metadata={"path": rel},
    )


def _read_parsed_document(
    root: Path, file_path: Path
) -> CorpusDocument:
    rel = str(file_path.relative_to(root))
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text, parser_metadata = parse_pdf_text(file_path)
    elif ext == ".docx":
        text, parser_metadata = parse_docx_text(file_path)
    else:
        raise ValueError(
            f"unsupported parsed document extension: {ext}"
        )

    return CorpusDocument(
        id=_stable_id(rel, text),
        text=text,
        source=rel,
        title=extract_title(text, file_path.stem),
        format=ext.lstrip("."),
        metadata={"path": rel, **parser_metadata},
    )


def _read_json_documents(
    root: Path, file_path: Path
) -> list[CorpusDocument]:
    try:
        raw = file_path.read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError as exc:
        warnings.warn(
            f"could not read JSON file {file_path}: {exc}",
            stacklevel=2,
        )
        return []

    rel = str(file_path.relative_to(root))
    if file_path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            raw.splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                warnings.warn(
                    f"skipping malformed JSONL line "
                    f"{line_number} in {file_path}: {exc}",
                    stacklevel=2,
                )
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"could not parse JSON file "
                f"{file_path}: {exc}",
                stacklevel=2,
            )
            return []
        records = parsed if isinstance(parsed, list) else [parsed]

    documents: list[CorpusDocument] = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            text = record
            metadata = {}
            title = ""
        elif isinstance(record, dict):
            text = _first_string(
                record,
                "page_content",
                "content",
                "text",
                "body",
                "document",
            )
            metadata = (
                record.get("metadata")
                if isinstance(record.get("metadata"), dict)
                else {}
            )
            title = str(
                record.get("title")
                or metadata.get("title")
                or ""
            )
        else:
            continue
        if not text or not text.strip():
            continue
        source = f"{rel}#{index}"
        documents.append(
            CorpusDocument(
                id=_stable_id(source, text),
                text=text,
                source=source,
                title=title or Path(rel).stem,
                format="json",
                metadata={"path": rel, **metadata},
            )
        )
    return documents


def _extract_markdown_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith(
            "## "
        ):
            return stripped[2:].strip()
    return ""


def _first_string(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _stable_id(source: str, text: str) -> str:
    digest = hashlib.sha256(
        f"{source}\0{text}".encode()
    ).hexdigest()[:16]
    return f"doc_{digest}"
