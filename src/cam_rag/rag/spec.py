"""Application profile contracts for reusable RAG apps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from cam_rag.rag.models import Chunk


class Tokenizer(Protocol):
    """Domain tokenizer hook used by sparse retrieval and coverage metrics."""

    def __call__(self, text: str) -> list[str]:
        """Tokenize text into searchable terms."""


def default_tokenizer(text: str) -> list[str]:
    """Simple tokenizer for generic app defaults and tests."""

    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9]+", text)]


@dataclass(slots=True)
class RAGPolicy:
    """Policy switches that apps can tighten without forking the platform."""

    enforce_phi: bool = False
    enforce_pii: bool = False
    use_llm_redaction: bool = False
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
    chunk_overlap: int = 0
    chunk_strategy: str = "paragraph"  # "paragraph" | "semantic" | "sentence"
    query_expansion_enabled: bool = False
    expansion_terms: int = 5
    reranker_model: str | None = None
    domain_tags: tuple[str, ...] = ()
    use_pipeline: bool = False
    embedding_backend: Any = None
    embedding_backends: list[Any] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    use_rl: bool = False
    rl_persistence_path: str | None = None
    use_governance: bool = False
    generation_backend: Any = None
    reranker_backend: Any = None
    multi_hop_enabled: bool = False
    multi_hop_max_hops: int = 2
    moe_scoring_enabled: bool = False
    accuracy_contracts_enabled: bool = False
    selective_filter_enabled: bool = False
    selective_filter_threshold: float = 0.55
    etf_verification_enabled: bool = False
    complexity_routing_enabled: bool = False
    instruction_prefix: str = ""
    adaptive_fusion_enabled: bool = False
    hyde_enabled: bool = False
    cross_encoder_model: str = ""
    pipeline_strategy: str = "auto"
    retriever_plugins: list[Any] = field(default_factory=list)
    auto_calibrate: bool = False
    rerank_blend_weight: float = 1.0

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer(text)

    def accepts_chunk(self, chunk: Chunk) -> bool:
        return bool(chunk.text.strip())
