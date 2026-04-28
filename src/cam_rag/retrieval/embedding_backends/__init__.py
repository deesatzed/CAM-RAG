"""Pluggable embedding backends for dense retrieval."""

from cam_rag.retrieval.embedding_backends.ollama import OllamaEmbeddingBackend
from cam_rag.retrieval.embedding_backends.openrouter import OpenRouterEmbeddingBackend

__all__ = [
    "OllamaEmbeddingBackend",
    "OpenRouterEmbeddingBackend",
]
