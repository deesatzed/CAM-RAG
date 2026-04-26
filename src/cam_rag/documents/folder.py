"""Folder-based document ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cam_rag.documents.parsers import extract_title, parse_docx_text, parse_pdf_text
from cam_rag.rag.models import CorpusDocument
from cam_rag.rag.spec import RAGAppSpec

TEXT_EXTENSIONS = {".md", ".txt"}
JSON_EXTENSIONS = {".json", ".jsonl"}
PARSER_EXTENSIONS = {".pdf", ".docx"}


def read_document_folder(path: str | Path, spec: RAGAppSpec) -> list[CorpusDocument]:
    """Read supported files from a folder into normalized platform documents.

    Supports text/markdown, common JSON export shapes, and optional PDF/DOCX
    parser backends.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"document folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"document folder is not a directory: {root}")

    supported = {ext.lower() for ext in spec.supported_extensions}
    documents: list[CorpusDocument] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in supported:
            continue
        if ext in TEXT_EXTENSIONS:
            document = _read_text_document(root, file_path)
            if document.text.strip():
                documents.append(document)
        elif ext in JSON_EXTENSIONS:
            documents.extend(_read_json_documents(root, file_path))
        elif ext in PARSER_EXTENSIONS:
            document = _read_parsed_document(root, file_path)
            if document.text.strip():
                documents.append(document)
    return documents


def _read_text_document(root: Path, file_path: Path) -> CorpusDocument:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    rel = str(file_path.relative_to(root))
    return CorpusDocument(
        id=_stable_id(rel, text),
        text=text,
        source=rel,
        title=_extract_markdown_title(text) or file_path.stem,
        format=file_path.suffix.lower().lstrip("."),
        metadata={"path": rel},
    )


def _read_parsed_document(root: Path, file_path: Path) -> CorpusDocument:
    rel = str(file_path.relative_to(root))
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text, parser_metadata = parse_pdf_text(file_path)
    elif ext == ".docx":
        text, parser_metadata = parse_docx_text(file_path)
    else:
        raise ValueError(f"unsupported parsed document extension: {ext}")

    return CorpusDocument(
        id=_stable_id(rel, text),
        text=text,
        source=rel,
        title=extract_title(text, file_path.stem),
        format=ext.lstrip("."),
        metadata={"path": rel, **parser_metadata},
    )


def _read_json_documents(root: Path, file_path: Path) -> list[CorpusDocument]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    rel = str(file_path.relative_to(root))
    if file_path.suffix.lower() == ".jsonl":
        records = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    else:
        parsed = json.loads(raw)
        records = parsed if isinstance(parsed, list) else [parsed]

    documents: list[CorpusDocument] = []
    for index, record in enumerate(records):
        if isinstance(record, str):
            text = record
            metadata = {}
            title = ""
        elif isinstance(record, dict):
            text = _first_string(record, "page_content", "content", "text", "body", "document")
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            title = str(record.get("title") or metadata.get("title") or "")
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
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""


def _first_string(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _stable_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"
