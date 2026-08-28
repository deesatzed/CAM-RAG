"""
SQLite persistence for FractalRAG indexes.

Saves/loads the entire FractalRAG state (documents, profiles, index entries,
derivatives, adapters) to a SQLite database for fast reload without re-embedding.

Usage:
    from cam_rag.retrieval.fractal.storage import save, load

    save("my_index.db", rag)
    rag2 = load("my_index.db", backend=SentenceTransformerEmbedding("BAAI/bge-m3"))
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .core import EmbeddingBackend
from .engine import FractalRAG, IndexEntry
from .profile import DocumentProfile

SCHEMA_VERSION = "2"


class DimensionMismatchError(Exception):
    """Raised when the backend's embedding dim doesn't match the stored dim."""
    pass


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _create_tables(cur: sqlite3.Cursor) -> None:
    cur.execute("DROP TABLE IF EXISTS metadata")
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("DROP TABLE IF EXISTS profiles")
    cur.execute("DROP TABLE IF EXISTS index_entries")
    cur.execute("DROP TABLE IF EXISTS derivatives")
    cur.execute("DROP TABLE IF EXISTS adapters")
    cur.execute("DROP TABLE IF EXISTS doc_metadata")
    cur.execute("DROP TABLE IF EXISTS doc_titles")

    cur.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            text TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE profiles (
            doc_id TEXT PRIMARY KEY,
            accuracy_importance TEXT NOT NULL,
            complexity_level TEXT NOT NULL,
            update_frequency TEXT NOT NULL,
            domain TEXT NOT NULL,
            likely_question_types TEXT NOT NULL,
            tolerance_for_hallucination TEXT NOT NULL,
            priority TEXT NOT NULL,
            mission_critical INTEGER NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE index_entries (
            id TEXT PRIMARY KEY,
            level INTEGER NOT NULL,
            vec BLOB NOT NULL,
            text TEXT NOT NULL,
            parent TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_index_entries_level ON index_entries(level)")
    cur.execute("""
        CREATE TABLE derivatives (
            entry_id TEXT PRIMARY KEY,
            d1 BLOB NOT NULL,
            d2 BLOB NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE adapters (
            doc_id TEXT PRIMARY KEY,
            vec BLOB NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE doc_metadata (
            doc_id TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE doc_titles (
            doc_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)


def save(path: str | Path, rag: FractalRAG) -> None:
    """Save entire FractalRAG state to a SQLite database.

    This is a full-replacement save: all tables are dropped and recreated.
    The operation is wrapped in a single transaction for atomicity.

    Args:
        path: File path for the SQLite database (created if doesn't exist).
        rag: The FractalRAG instance to persist.
    """
    path = Path(path)
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        _create_tables(cur)

        # Metadata
        cur.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", SCHEMA_VERSION),
                ("embedding_dim", str(rag.dim)),
                ("backend_type", type(rag._backend).__name__),
                ("adapter_strength", str(rag._adapter_strength)),
            ],
        )

        # Documents
        cur.executemany(
            "INSERT INTO documents (doc_id, text) VALUES (?, ?)",
            list(rag.docs.items()),
        )

        # Profiles
        for doc_id, prof in rag.profiles.items():
            cur.execute(
                "INSERT INTO profiles (doc_id, accuracy_importance, complexity_level, "
                "update_frequency, domain, likely_question_types, "
                "tolerance_for_hallucination, priority, mission_critical) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    prof.accuracy_importance,
                    prof.complexity_level,
                    prof.update_frequency,
                    prof.domain,
                    json.dumps(prof.likely_question_types),
                    prof.tolerance_for_hallucination,
                    prof.priority,
                    int(prof.mission_critical),
                ),
            )

        # Index entries (preserve insertion order via rowid)
        for level in [0, 1, 2]:
            for entry in rag.index[level]:
                cur.execute(
                    "INSERT INTO index_entries (id, level, vec, text, parent) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        entry.level,
                        _vec_to_blob(entry.vec),
                        entry.text,
                        entry.parent,
                    ),
                )

        # Derivatives
        for entry_id, derivs in rag.derivatives.items():
            cur.execute(
                "INSERT INTO derivatives (entry_id, d1, d2) VALUES (?, ?, ?)",
                (entry_id, _vec_to_blob(derivs["d1"]), _vec_to_blob(derivs["d2"])),
            )

        # Adapters
        for doc_id, vec in rag.adapters.items():
            cur.execute(
                "INSERT INTO adapters (doc_id, vec) VALUES (?, ?)",
                (doc_id, _vec_to_blob(vec)),
            )

        # Document metadata
        for doc_id, meta in rag.doc_metadata.items():
            cur.execute(
                "INSERT INTO doc_metadata (doc_id, metadata_json) VALUES (?, ?)",
                (doc_id, json.dumps(meta, ensure_ascii=False)),
            )

        # Document titles
        for doc_id, title in rag.doc_titles.items():
            if title is not None:
                cur.execute(
                    "INSERT INTO doc_titles (doc_id, title) VALUES (?, ?)",
                    (doc_id, title),
                )

        conn.commit()
    finally:
        conn.close()


def load(path: str | Path, backend: EmbeddingBackend) -> FractalRAG:
    """Load a FractalRAG instance from a SQLite database.

    The user must provide an embedding backend. Its dimension is validated
    against the stored dimension to prevent silent corruption.

    Args:
        path: Path to an existing SQLite database created by save().
        backend: Embedding backend to attach to the loaded engine.
            Must have the same dim as the one used when saving.

    Returns:
        A fully reconstructed FractalRAG instance with all state restored.

    Raises:
        FileNotFoundError: If path doesn't exist.
        DimensionMismatchError: If backend.dim != stored dim.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index database not found: {path}")

    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()

        # Read metadata
        meta = dict(cur.execute("SELECT key, value FROM metadata").fetchall())
        stored_dim = int(meta["embedding_dim"])
        adapter_strength = float(meta["adapter_strength"])

        if backend.dim != stored_dim:
            raise DimensionMismatchError(
                f"Backend dim={backend.dim} does not match stored dim={stored_dim}"
            )

        rag = FractalRAG(backend=backend, adapter_strength=adapter_strength)

        # Documents
        for doc_id, text in cur.execute("SELECT doc_id, text FROM documents"):
            rag.docs[doc_id] = text

        # Profiles
        for row in cur.execute(
            "SELECT doc_id, accuracy_importance, complexity_level, update_frequency, "
            "domain, likely_question_types, tolerance_for_hallucination, priority, "
            "mission_critical FROM profiles"
        ):
            rag.profiles[row[0]] = DocumentProfile(
                accuracy_importance=row[1],
                complexity_level=row[2],
                update_frequency=row[3],
                domain=row[4],
                likely_question_types=json.loads(row[5]),
                tolerance_for_hallucination=row[6],
                priority=row[7],
                mission_critical=bool(row[8]),
            )

        # Index entries (ORDER BY level, rowid preserves insertion order)
        for row in cur.execute(
            "SELECT id, level, vec, text, parent FROM index_entries "
            "ORDER BY level, rowid"
        ):
            entry = IndexEntry(
                id=row[0],
                level=row[1],
                vec=_blob_to_vec(row[2]),
                text=row[3],
                parent=row[4],
            )
            rag.index[entry.level].append(entry)

        # Derivatives
        for entry_id, d1_blob, d2_blob in cur.execute(
            "SELECT entry_id, d1, d2 FROM derivatives"
        ):
            rag.derivatives[entry_id] = {
                "d1": _blob_to_vec(d1_blob),
                "d2": _blob_to_vec(d2_blob),
            }

        # Adapters
        for doc_id, vec_blob in cur.execute("SELECT doc_id, vec FROM adapters"):
            rag.adapters[doc_id] = _blob_to_vec(vec_blob)

        # Document metadata (backward-compatible: old DBs may not have this table)
        try:
            for doc_id, meta_json in cur.execute(
                "SELECT doc_id, metadata_json FROM doc_metadata"
            ):
                rag.doc_metadata[doc_id] = json.loads(meta_json)
        except sqlite3.OperationalError:
            pass  # Table doesn't exist in old schema — that's fine

        # Document titles (backward-compatible)
        try:
            for doc_id, title in cur.execute(
                "SELECT doc_id, title FROM doc_titles"
            ):
                rag.doc_titles[doc_id] = title
        except sqlite3.OperationalError:
            pass  # Table doesn't exist in old schema — that's fine

        return rag
    finally:
        conn.close()
