"""Contrastive training data generation from corpus chunks.

Generates (anchor, positive, negative) triplets without any LLM calls
by exploiting document structure and BM25 near-misses.

Strategies
----------
1. **Title-body pairs**: chunk heading/title as anchor, body as positive.
2. **Adjacent pairs**: consecutive chunks from the same document as (anchor, positive).
3. **Hard negatives**: BM25 near-misses from a different document.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any

from cam_rag.rag.models import Chunk
from cam_rag.retrieval.models import RetrievalDocument
from cam_rag.retrieval.sparse import SparseBM25Retriever

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate contrastive triplets from corpus chunks.

    Parameters
    ----------
    chunks:
        All corpus chunks to generate training data from.
    seed:
        Random seed for reproducibility.
    max_negatives_per_query:
        Maximum BM25 candidates to consider for hard-negative selection.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        seed: int = 42,
        max_negatives_per_query: int = 10,
    ) -> None:
        self.chunks = list(chunks)
        self._seed = seed
        self._max_neg = max_negatives_per_query
        self._rng = random.Random(seed)
        self._by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in self.chunks:
            self._by_doc[c.document_id].append(c)
        # Sort by position within each document
        for doc_id in self._by_doc:
            self._by_doc[doc_id].sort(key=lambda c: c.position)

    def generate(self) -> list[dict[str, str]]:
        """Return a list of ``{"anchor", "positive", "negative"}`` dicts.

        Each dict represents one contrastive training triplet.
        """
        title_body = self._title_body_pairs()
        adjacent = self._adjacent_pairs()
        positives = title_body + adjacent

        if not positives:
            logger.warning("No positive pairs generated; returning empty dataset")
            return []

        # Build BM25 index for hard-negative mining
        bm25 = self._build_bm25()
        triplets = self._attach_hard_negatives(positives, bm25)

        self._rng.shuffle(triplets)
        logger.info(
            "generated %d triplets (%d title-body, %d adjacent)",
            len(triplets),
            len(title_body),
            len(adjacent),
        )
        return triplets

    def generate_dataset(self) -> Any:
        """Return a HuggingFace ``Dataset`` with anchor/positive/negative columns.

        Requires the ``datasets`` package to be installed.
        """
        triplets = self.generate()
        if not triplets:
            try:
                from datasets import Dataset

                return Dataset.from_dict({"anchor": [], "positive": [], "negative": []})
            except ImportError as exc:
                raise ImportError(
                    "datasets is required. Install with: pip install datasets"
                ) from exc

        from datasets import Dataset

        return Dataset.from_dict(
            {
                "anchor": [t["anchor"] for t in triplets],
                "positive": [t["positive"] for t in triplets],
                "negative": [t["negative"] for t in triplets],
            }
        )

    # ------------------------------------------------------------------
    # Pair generation strategies
    # ------------------------------------------------------------------

    def _title_body_pairs(self) -> list[dict[str, str]]:
        """Strategy 1: Use chunk heading/title as anchor, body text as positive."""
        pairs: list[dict[str, str]] = []
        for chunk in self.chunks:
            heading = chunk.section_heading or chunk.title
            if not heading or not heading.strip():
                continue
            # Skip if body is too short
            body = chunk.text.strip()
            if len(body.split()) < 5:
                continue
            pairs.append({"anchor": heading.strip(), "positive": body, "doc_id": chunk.document_id})
        return pairs

    def _adjacent_pairs(self) -> list[dict[str, str]]:
        """Strategy 2: Consecutive chunks from same document share topic."""
        pairs: list[dict[str, str]] = []
        for doc_id, doc_chunks in self._by_doc.items():
            if len(doc_chunks) < 2:
                continue
            for i in range(len(doc_chunks) - 1):
                a = doc_chunks[i].text.strip()
                b = doc_chunks[i + 1].text.strip()
                if len(a.split()) < 5 or len(b.split()) < 5:
                    continue
                pairs.append({"anchor": a, "positive": b, "doc_id": doc_id})
        return pairs

    # ------------------------------------------------------------------
    # Hard-negative mining
    # ------------------------------------------------------------------

    def _build_bm25(self) -> SparseBM25Retriever:
        """Build a BM25 index over all chunks."""
        docs = [
            RetrievalDocument(
                doc_id=c.id,
                text=c.text,
                metadata={"document_id": c.document_id},
            )
            for c in self.chunks
        ]
        return SparseBM25Retriever(docs)

    def _attach_hard_negatives(
        self,
        positives: list[dict[str, str]],
        bm25: SparseBM25Retriever,
    ) -> list[dict[str, str]]:
        """For each (anchor, positive) pair, find a hard negative via BM25.

        A hard negative is a high-BM25-scoring chunk from a *different*
        document than the positive — it has high lexical overlap but is
        about a different topic.
        """
        # Build a lookup: chunk_id -> document_id
        chunk_doc_map = {c.id: c.document_id for c in self.chunks}
        chunk_text_map = {c.id: c.text for c in self.chunks}

        # Collect all chunk texts for fallback random negatives
        all_texts_by_doc: dict[str, list[str]] = defaultdict(list)
        for c in self.chunks:
            all_texts_by_doc[c.document_id].append(c.text)

        triplets: list[dict[str, str]] = []
        for pair in positives:
            anchor = pair["anchor"]
            doc_id = pair.get("doc_id", "")

            # BM25 search using anchor as query
            results = bm25.retrieve(anchor, k=self._max_neg)
            hard_neg = None
            for result in results:
                result_doc_id = result.metadata.get("document_id", "")
                if result_doc_id != doc_id:
                    hard_neg = result.text
                    break

            if hard_neg is None:
                # Fallback: random text from a different document
                hard_neg = self._random_negative(doc_id, all_texts_by_doc)

            if hard_neg is not None:
                triplets.append(
                    {
                        "anchor": anchor,
                        "positive": pair["positive"],
                        "negative": hard_neg,
                    }
                )

        return triplets

    def _random_negative(
        self,
        exclude_doc_id: str,
        all_texts_by_doc: dict[str, list[str]],
    ) -> str | None:
        """Pick a random text from a document other than *exclude_doc_id*."""
        other_docs = [did for did in all_texts_by_doc if did != exclude_doc_id]
        if not other_docs:
            return None
        doc = self._rng.choice(other_docs)
        return self._rng.choice(all_texts_by_doc[doc])
