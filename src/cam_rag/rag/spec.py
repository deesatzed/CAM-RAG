"""Application profile contracts for reusable RAG apps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cam_rag.rag.models import Chunk


class Tokenizer(Protocol):
    """Domain tokenizer hook used by sparse retrieval and coverage metrics."""

    def __call__(self, text: str) -> list[str]:
        """Tokenize text into searchable terms."""


def default_tokenizer(text: str) -> list[str]:
    """Simple tokenizer for generic app defaults and tests."""

    return [part.lower() for part in text.split() if part.strip()]


@dataclass(slots=True)
class RAGPolicy:
    """Policy switches that apps can tighten without forking the platform."""

    enforce_phi: bool = False
    enforce_pii: bool = False
    require_citations: bool = True
    min_confidence: float = 0.4
    allowed_residency: str | None = None


@dataclass(slots=True)
class RAGAppSpec:
    """Declarative profile for an application built on the platform."""

    name: str
    description: str = ""
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".pdf", ".docx")
    tokenizer: Tokenizer = default_tokenizer
    policy: RAGPolicy = field(default_factory=RAGPolicy)
    retrieval_top_k: int = 10
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    reranker_model: str | None = None
    domain_tags: tuple[str, ...] = ()

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer(text)

    def accepts_chunk(self, chunk: Chunk) -> bool:
        return bool(chunk.text.strip())
