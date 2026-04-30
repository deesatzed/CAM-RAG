"""Pluggable retrieval backend protocol for extensible pipeline fusion.

New retrieval methods (SPLADE, ColBERT, learned sparse, etc.) can implement
the ``RetrieverPlugin`` protocol and be added to the pipeline via
``RAGAppSpec.retriever_plugins``.  Plugins participate in N-way RRF fusion
alongside the built-in BM25 and dense retrievers.

Example usage::

    class SPLADEPlugin:
        name = "splade"

        def index(self, documents):
            self._index = build_splade_index(documents)

        def retrieve(self, query, k=10):
            return self._index.search(query, k)

    spec = RAGAppSpec(
        name="my-app",
        use_pipeline=True,
        retriever_plugins=[SPLADEPlugin()],
    )
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cam_rag.retrieval.models import RetrievalDocument, RetrievalResult


@runtime_checkable
class RetrieverPlugin(Protocol):
    """Protocol for pluggable retrieval backends.

    Any class implementing this protocol can be registered with the
    pipeline and participate in N-way RRF fusion alongside BM25
    and dense vector retrieval.
    """

    @property
    def name(self) -> str:
        """Unique name for this retriever (used in RRF weight keys and logs)."""
        ...

    def index(self, documents: list[RetrievalDocument]) -> None:
        """Build internal index from documents.  Called once at setup time."""
        ...

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-k documents for a query."""
        ...


class PluginRegistry:
    """Registry for retriever plugins.

    Provides named lookup and iteration over registered plugins.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, RetrieverPlugin] = {}

    def register(self, plugin: RetrieverPlugin) -> None:
        """Register a retriever plugin by its ``name``."""
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> RetrieverPlugin:
        """Look up a plugin by name. Raises ``KeyError`` if not found."""
        if name not in self._plugins:
            raise KeyError(f"No retriever plugin registered: {name}")
        return self._plugins[name]

    @property
    def all(self) -> list[RetrieverPlugin]:
        """All registered plugins in insertion order."""
        return list(self._plugins.values())

    @property
    def names(self) -> list[str]:
        """Names of all registered plugins."""
        return list(self._plugins.keys())

    def __len__(self) -> int:
        return len(self._plugins)
