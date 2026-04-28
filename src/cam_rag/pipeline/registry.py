"""Technique registry and pipeline composer.

The registry holds metadata about available retrieval/ranking techniques.
``compose_pipeline`` performs topological sorting with query-type filtering
to produce an ``ExecutionPlan`` -- an ordered list of techniques to execute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TechniqueDescriptor:
    """Metadata for one registered retrieval/ranking technique."""

    name: str
    category: str  # "retrieval", "reranking", "expansion", "verification"
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    query_types: tuple[str, ...] = (
        "specification",
        "summary",
        "logic",
        "synthesis",
    )
    min_corpus_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """An ordered list of techniques to execute for one query."""

    steps: tuple[str, ...]
    query_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TechniqueRegistry:
    """Registry of available retrieval/ranking techniques.

    Techniques must be registered in dependency order: every technique
    listed in ``requires`` must already be present in the registry.
    """

    _techniques: dict[str, TechniqueDescriptor] = field(default_factory=dict)

    def register(self, descriptor: TechniqueDescriptor) -> None:
        """Register a technique, validating its dependencies exist."""
        if descriptor.name in self._techniques:
            raise ValueError(f"Technique already registered: {descriptor.name}")
        for dep in descriptor.requires:
            if dep not in self._techniques:
                raise ValueError(
                    f"Technique {descriptor.name} requires unregistered technique: {dep}"
                )
        self._techniques[descriptor.name] = descriptor

    def get(self, name: str) -> TechniqueDescriptor:
        """Look up a technique by name."""
        if name not in self._techniques:
            raise KeyError(f"Technique not registered: {name}")
        return self._techniques[name]

    def list_for_query_type(self, query_type: str) -> list[TechniqueDescriptor]:
        """Return all techniques applicable to *query_type*."""
        return [
            t
            for t in self._techniques.values()
            if query_type in t.query_types
        ]

    @property
    def names(self) -> set[str]:
        """Set of all registered technique names."""
        return set(self._techniques)


def compose_pipeline(
    registry: TechniqueRegistry,
    query_type: str,
    *,
    corpus_size: int = 10,
    enabled_overrides: set[str] | None = None,
) -> ExecutionPlan:
    """Compose an execution plan from available techniques.

    Performs topological sorting with query-type filtering and optional
    technique subsetting via *enabled_overrides*.
    """
    available = registry.list_for_query_type(query_type)
    available_names = {t.name for t in available}

    if enabled_overrides is not None:
        available_names &= enabled_overrides

    # Filter by corpus size
    available = [
        t for t in available if t.name in available_names and t.min_corpus_size <= corpus_size
    ]
    available_names = {t.name for t in available}

    # Topological sort respecting requires
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        desc = registry.get(name)
        for dep in desc.requires:
            if dep in available_names:
                visit(dep)
        ordered.append(name)

    for t in available:
        visit(t.name)

    return ExecutionPlan(steps=tuple(ordered), query_type=query_type)
