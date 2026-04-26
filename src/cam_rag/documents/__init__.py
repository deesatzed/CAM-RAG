"""Document ingestion helpers for the RAG platform."""

from cam_rag.documents.chunking import chunk_document, chunk_documents
from cam_rag.documents.folder import read_document_folder

__all__ = ["chunk_document", "chunk_documents", "read_document_folder"]
