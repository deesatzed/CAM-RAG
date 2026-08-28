"""
FractalRAG — Unified fractal multi-scale retrieval engine.

Consolidates FractalLatentRAG (v1), FractalSOTARAG (v2),
FractalSOTARAGv3 (v3), and xAIKnowledgeEngine into a single class.

All features are preserved. Profile-driven features are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .core import EmbeddingBackend, HashEmbedding, chunk_fractal, normalize
from .profile import DocumentProfile
from .query import BASE_DERIV_WEIGHT, classify_query_type, extract_domain_hints, get_type_weights


@dataclass
class IndexEntry:
    """Typed index entry replacing raw dicts."""
    id: str
    level: int       # 0=sentence, 1=paragraph, 2=document
    vec: np.ndarray
    text: str
    parent: str | None = None


class FractalRAG:
    """Unified Fractal Latent RAG engine.

    Modes:
        Basic (v1):     FractalRAG(backend).add_document(id, text) / .retrieve(query)
        Adaptive (v2):  Same, but retrieve auto-classifies query and uses type-aware weights.
        Full (v3):      add_document(..., profile=DocumentProfile(...)) for profile-driven behavior.
        Flat baseline:  retrieve(query, levels=[2]) for doc-level-only retrieval (benchmark).
    """

    def __init__(
        self,
        backend: EmbeddingBackend | None = None,
        adapter_strength: float = 0.25,
    ):
        self._backend = backend or HashEmbedding(dim=64)
        self._adapter_strength = adapter_strength
        self.docs: dict[str, str] = {}
        self.profiles: dict[str, DocumentProfile] = {}
        self.adapters: dict[str, np.ndarray] = {}
        self.index: dict[int, list[IndexEntry]] = {0: [], 1: [], 2: []}
        self.derivatives: dict[str, dict[str, np.ndarray]] = {}
        self.doc_metadata: dict[str, dict] = {}
        self.doc_titles: dict[str, str | None] = {}

    @property
    def dim(self) -> int:
        return self._backend.dim

    def _embed(self, text: str) -> np.ndarray:
        vec = self._backend.embed(text)
        return np.asarray(vec, dtype=np.float32)

    def add_document(
        self,
        doc_id: str,
        text: str,
        profile: DocumentProfile | None = None,
        learn_adapter: bool = True,
        metadata: dict | None = None,
        title: str | None = None,
    ) -> None:
        """Index a document at all fractal scales with optional profile-driven config.

        Args:
            title: Optional document title. When provided, prepended to L0/L1
                   embeddings as "{title}: {text}" for contextual enrichment.
        """
        self.docs[doc_id] = text
        if profile:
            self.profiles[doc_id] = profile
        if metadata is not None:
            self.doc_metadata[doc_id] = metadata
        self.doc_titles[doc_id] = title

        config = profile.to_config() if profile else None
        strength = config["adapter_strength"] if config else self._adapter_strength
        max_sentences = (config["deriv_depth"] * 4) if config else None

        sentences, paragraphs, full_text = chunk_fractal(text)

        # Level 2: Document
        doc_vec = self._embed(full_text)
        if learn_adapter:
            adapter_seed = f"ADAPTER_{doc_id}"
            if profile:
                adapter_seed += f"_{profile.domain}"
            adapter = self._embed(adapter_seed) * strength
            self.adapters[doc_id] = adapter
            doc_vec = normalize(doc_vec + adapter)
        else:
            doc_vec = normalize(doc_vec)
            adapter = np.zeros(self.dim)

        self.index[2].append(IndexEntry(
            id=doc_id, level=2, vec=doc_vec,
            text=full_text[:220] if len(full_text) > 220 else full_text,
        ))

        # Level 1: Paragraphs
        adapter = self.adapters.get(doc_id, np.zeros(self.dim))
        para_vecs = []
        for i, para in enumerate(paragraphs):
            embed_text = f"{title}: {para}" if title else para
            pvec = normalize(self._embed(embed_text) + adapter * 0.8)
            self.index[1].append(IndexEntry(
                id=f"{doc_id}_p{i}", level=1, vec=pvec,
                text=para[:160] if len(para) > 160 else para,
                parent=doc_id,
            ))
            para_vecs.append(pvec)

        # Weighted paragraph mean: weight by similarity to doc_vec
        if para_vecs:
            weights = np.array([max(0.1, float(np.dot(pv, doc_vec))) for pv in para_vecs])
            weights = weights / weights.sum()
            para_mean = sum(w * pv for w, pv in zip(weights, para_vecs, strict=True))
        else:
            para_mean = np.zeros(self.dim)

        # Level 0: Sentences + Derivatives
        sent_list = sentences[:max_sentences] if max_sentences else sentences
        for i, sent in enumerate(sent_list):
            embed_text = f"{title}: {sent}" if title else sent
            svec = normalize(self._embed(embed_text) + adapter * 0.6)
            sid = f"{doc_id}_s{i}"
            self.index[0].append(IndexEntry(
                id=sid, level=0, vec=svec, text=sent, parent=doc_id,
            ))
            # 1st derivative: delta from paragraph mean (knowledge velocity)
            d1 = normalize(svec - para_mean)
            # 2nd derivative: curvature (acceleration of concepts)
            d2 = normalize(d1 - (svec - doc_vec))
            self.derivatives[sid] = {"d1": d1, "d2": d2}

    def _get_doc_id(self, entry: IndexEntry) -> str:
        """Get the parent doc_id for any index entry."""
        return entry.parent if entry.parent else entry.id

    def _filter_by_metadata(self, doc_id: str, filters: dict) -> bool:
        """Check if a document passes all metadata filters.

        Returns True if the document passes (should be included).
        Documents without metadata fail all filters.
        """
        meta = self.doc_metadata.get(doc_id)
        if meta is None:
            return False

        for key, value in filters.items():
            if key == "domain":
                doc_domain = meta.get("domain")
                if isinstance(value, list):
                    if doc_domain not in value:
                        return False
                else:
                    if doc_domain != value:
                        return False

            elif key == "year_range":
                doc_year = meta.get("year")
                if doc_year is None:
                    return False
                if not (value[0] <= doc_year <= value[1]):
                    return False

            elif key == "year_min":
                doc_year = meta.get("year")
                if doc_year is None or doc_year < value:
                    return False

            elif key == "year_max":
                doc_year = meta.get("year")
                if doc_year is None or doc_year > value:
                    return False

            elif key == "mesh_terms":
                doc_mesh = meta.get("mesh_terms", [])
                if not any(t in doc_mesh for t in value):
                    return False

            elif key == "journal":
                doc_journal = meta.get("journal")
                if isinstance(value, list):
                    if doc_journal not in value:
                        return False
                else:
                    if doc_journal != value:
                        return False

        return True

    def _compute_metadata_boost(self, doc_id: str, boost_config: dict) -> float:
        """Compute additive metadata boost for a document.

        Boost keys:
            domain_boost + domain_target: flat boost for matching domain
            mesh_boost + mesh_target: additive per matching MeSH term
            recency_boost: linear by year (2000=0, 2026=1)
        """
        meta = self.doc_metadata.get(doc_id)
        if meta is None:
            return 0.0

        boost = 0.0

        # Domain boost
        if "domain_boost" in boost_config and "domain_target" in boost_config:
            doc_domain = meta.get("domain")
            target = boost_config["domain_target"]
            if isinstance(target, list):
                if doc_domain in target:
                    boost += boost_config["domain_boost"]
            else:
                if doc_domain == target:
                    boost += boost_config["domain_boost"]

        # MeSH term boost (additive per matching term)
        if "mesh_boost" in boost_config and "mesh_target" in boost_config:
            doc_mesh = set(meta.get("mesh_terms", []))
            target_mesh = set(boost_config["mesh_target"])
            n_matches = len(doc_mesh & target_mesh)
            boost += n_matches * boost_config["mesh_boost"]

        # Recency boost (linear: 2000=0, 2026=1)
        if "recency_boost" in boost_config:
            doc_year = meta.get("year")
            if doc_year is not None:
                recency = max(0.0, min(1.0, (doc_year - 2000) / 26.0))
                boost += recency * boost_config["recency_boost"]

        return boost

    def retrieve(
        self,
        query: str,
        k: int = 3,
        levels: list[int] | None = None,
        query_type: str | None = None,
        profile: DocumentProfile | None = None,
        focus_doc: str | None = None,
        use_derivatives: bool = True,
        use_level_weights: bool = True,
        metadata_filters: dict | None = None,
    ) -> tuple[dict[int, list[tuple[IndexEntry, float]]], str]:
        """Multi-scale retrieval with configurable features.

        Args:
            query: Query text.
            k: Results per level (overridden by profile if provided).
            levels: Which levels to search. Default [2,1,0] (all).
                    Use [2] for flat doc-only baseline.
            query_type: Override auto-classification.
            profile: DocumentProfile for adaptive behavior.
            focus_doc: Filter results to one document.
            use_derivatives: Whether to add derivative scoring bonus.
            use_level_weights: Whether to weight scores by level (type-aware).

        Returns:
            (results_by_level, detected_query_type)
        """
        if levels is None:
            levels = [2, 1, 0]

        qtype = query_type or classify_query_type(query, profile)
        weights = get_type_weights(qtype, profile) if use_level_weights else None
        effective_k = k
        if weights:
            effective_k = weights.get("k_per_level", k)

        qvec = self._embed(query)
        deriv_mult = weights["deriv_mult"] if weights else 1.0

        results = {}
        for lvl in levels:
            candidates = self.index[lvl]
            if focus_doc:
                candidates = [c for c in candidates
                              if (c.parent == focus_doc) or (c.id == focus_doc)]
            if metadata_filters:
                candidates = [c for c in candidates
                              if self._filter_by_metadata(self._get_doc_id(c), metadata_filters)]

            scored = []
            for item in candidates:
                base_sim = float(np.dot(qvec, item.vec))

                # Level weighting
                if use_level_weights and weights:
                    lvl_idx = [2, 1, 0].index(lvl)
                    base_sim *= weights["levels"][lvl_idx]

                # Derivative bonus
                deriv_bonus = 0.0
                if use_derivatives and item.id in self.derivatives:
                    d = self.derivatives[item.id]
                    deriv_bonus = (
                        float(np.dot(qvec, d["d1"])) * BASE_DERIV_WEIGHT +
                        float(np.dot(qvec, d["d2"])) * BASE_DERIV_WEIGHT * 0.6
                    ) * deriv_mult

                scored.append((item, base_sim + deriv_bonus))

            scored.sort(key=lambda x: x[1], reverse=True)
            results[lvl] = scored[:effective_k]

        return results, qtype

    def retrieve_reranked(
        self,
        query: str,
        k: int = 5,
        query_type: str | None = None,
        profile: DocumentProfile | None = None,
        focus_doc: str | None = None,
        use_derivatives: bool = True,
        para_boost: float = 0.15,
        sent_boost: float = 0.10,
        deriv_boost: float = 0.05,
        metadata_filters: dict | None = None,
        metadata_boost: dict | None = None,
        diversity_boost: float = 0.0,
    ) -> tuple[dict[int, list[tuple[IndexEntry, float]]], str]:
        """Reranked fractal retrieval: doc-level primary, sub-doc boosts.

        Strategy:
            1. Score all documents at Level 2 (primary ranking signal).
            2. Score paragraphs (L1) and sentences (L0) independently.
            3. For each document, compute boost from its best sub-doc matches.
            4. Final score = doc_score + para_boost * best_para_sim + sent_boost * best_sent_sim
               + optional derivative bonus from best matching sentence.
            5. Return reranked results in the same format as retrieve().

        This preserves the recall advantage (multi-scale finds more relevant docs)
        while fixing ranking (doc-level anchors the score, sub-doc evidence boosts).
        """
        qtype = query_type or classify_query_type(query, profile)
        qvec = self._embed(query)

        # Step 1: Score all documents at Level 2
        doc_scores: dict[str, float] = {}
        doc_entries: dict[str, IndexEntry] = {}
        for item in self.index[2]:
            if focus_doc and item.id != focus_doc:
                continue
            if metadata_filters and not self._filter_by_metadata(item.id, metadata_filters):
                continue
            sim = float(np.dot(qvec, item.vec))
            doc_scores[item.id] = sim
            doc_entries[item.id] = item

        # Step 2: Score paragraphs — track best per parent doc
        para_best: dict[str, float] = {}
        para_entries: dict[str, list[tuple[IndexEntry, float]]] = {}
        for item in self.index[1]:
            parent = item.parent or item.id
            if focus_doc and parent != focus_doc:
                continue
            if metadata_filters and not self._filter_by_metadata(parent, metadata_filters):
                continue
            sim = float(np.dot(qvec, item.vec))
            if parent not in para_best or sim > para_best[parent]:
                para_best[parent] = sim
            para_entries.setdefault(parent, []).append((item, sim))

        # Step 3: Score sentences — track best per parent doc + derivative bonus
        sent_best: dict[str, float] = {}
        sent_deriv_best: dict[str, float] = {}
        sent_entries: dict[str, list[tuple[IndexEntry, float]]] = {}
        for item in self.index[0]:
            parent = item.parent or item.id
            if focus_doc and parent != focus_doc:
                continue
            if metadata_filters and not self._filter_by_metadata(parent, metadata_filters):
                continue
            sim = float(np.dot(qvec, item.vec))
            if parent not in sent_best or sim > sent_best[parent]:
                sent_best[parent] = sim
            sent_entries.setdefault(parent, []).append((item, sim))

            # Derivative bonus for this sentence
            if use_derivatives and item.id in self.derivatives:
                d = self.derivatives[item.id]
                d_bonus = (
                    float(np.dot(qvec, d["d1"])) * BASE_DERIV_WEIGHT +
                    float(np.dot(qvec, d["d2"])) * BASE_DERIV_WEIGHT * 0.6
                )
                if parent not in sent_deriv_best or d_bonus > sent_deriv_best[parent]:
                    sent_deriv_best[parent] = d_bonus

        # Step 4: Combine — doc score is primary, sub-doc is boost
        # Also discover docs found ONLY at sub-doc level (recall expansion)
        all_doc_ids = set(doc_scores.keys()) | set(para_best.keys()) | set(sent_best.keys())

        # Compute score statistics for adaptive thresholding
        doc_score_vals = list(doc_scores.values())
        doc_mean = np.mean(doc_score_vals) if doc_score_vals else 0.0
        doc_std = np.std(doc_score_vals) if doc_score_vals else 1.0

        combined: list[tuple[str, float]] = []
        for doc_id in all_doc_ids:
            base = doc_scores.get(doc_id, 0.0)
            p_boost = para_best.get(doc_id, 0.0) * para_boost
            s_boost = sent_best.get(doc_id, 0.0) * sent_boost
            d_boost = sent_deriv_best.get(doc_id, 0.0) * deriv_boost if use_derivatives else 0.0

            # If doc wasn't in L2 index but found via sub-doc, use sub-doc as base
            # Adaptive discount: strong sub-doc matches get less discount
            if doc_id not in doc_scores:
                best_sub = max(para_best.get(doc_id, 0.0), sent_best.get(doc_id, 0.0))
                # If sub-doc match is above doc-mean, it's likely relevant
                # mild discount if strong sub-doc evidence, else heavy discount
                base = best_sub * 0.7 if best_sub > doc_mean + 0.5 * doc_std else best_sub * 0.4

            m_boost = (
                self._compute_metadata_boost(doc_id, metadata_boost)
                if metadata_boost else 0.0
            )
            final_score = base + p_boost + s_boost + d_boost + m_boost
            combined.append((doc_id, final_score))

        combined.sort(key=lambda x: x[1], reverse=True)

        # Domain-based MMR for diversity (synthesis queries)
        if diversity_boost > 0 and len(combined) > k:
            combined = self._apply_domain_diversity(combined, k, diversity_boost)

        # Step 5: Build results in the standard format
        results: dict[int, list[tuple[IndexEntry, float]]] = {2: [], 1: [], 0: []}
        seen_docs = set()
        for doc_id, score in combined[:k]:
            seen_docs.add(doc_id)
            # Level 2 entry
            if doc_id in doc_entries:
                results[2].append((doc_entries[doc_id], score))

            # Best paragraph for this doc
            if doc_id in para_entries:
                para_sorted = sorted(para_entries[doc_id], key=lambda x: x[1], reverse=True)
                results[1].append(para_sorted[0])

            # Best sentence for this doc
            if doc_id in sent_entries:
                sent_sorted = sorted(sent_entries[doc_id], key=lambda x: x[1], reverse=True)
                results[0].append(sent_sorted[0])

        return results, qtype

    def _apply_domain_diversity(
        self,
        scored_docs: list[tuple[str, float]],
        k: int,
        lambda_diversity: float,
    ) -> list[tuple[str, float]]:
        """Greedy domain-aware MMR reranking.

        At each step, select the doc that maximizes:
            score = (1 - lambda) * relevance + lambda * domain_novelty

        domain_novelty = 1.0 if doc's domain not yet in selected set, else 0.0
        """
        selected: list[tuple[str, float]] = []
        selected_domains: set = set()
        remaining = list(scored_docs)

        # Normalize relevance scores to [0, 1] for fair combination
        if remaining:
            max_score = remaining[0][1]
            min_score = remaining[-1][1]
            score_range = max_score - min_score if max_score > min_score else 1.0

        while len(selected) < k and remaining:
            best_idx = 0
            best_mmr = -float('inf')
            for i, (doc_id, raw_score) in enumerate(remaining):
                # Normalize relevance
                rel = (raw_score - min_score) / score_range if score_range > 0 else 1.0

                # Domain novelty
                doc_domain = self.doc_metadata.get(doc_id, {}).get("domain")
                novelty = 1.0 if doc_domain and doc_domain not in selected_domains else 0.0

                mmr = (1.0 - lambda_diversity) * rel + lambda_diversity * novelty
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            doc_id, score = remaining.pop(best_idx)
            selected.append((doc_id, score))
            doc_domain = self.doc_metadata.get(doc_id, {}).get("domain")
            if doc_domain:
                selected_domains.add(doc_domain)

        return selected

    # Default boost parameters per query type (tuned on medical corpus benchmark)
    # Post-3A: title-prefix absorbed sentence signal (sent_boost → 0),
    # deriv_boost increased in importance. Per-type tuning from Phase 3B.
    _TYPE_BOOSTS: ClassVar[dict[str, dict[str, float]]] = {
        "specification": {
            "para_boost": 0.0, "sent_boost": 0.0, "deriv_boost": 0.0, "diversity_boost": 0.0,
        },
        "summary": {
            "para_boost": 0.05, "sent_boost": 0.0, "deriv_boost": 0.10, "diversity_boost": 0.0,
        },
        "logic": {
            "para_boost": 0.0, "sent_boost": 0.0, "deriv_boost": 0.15, "diversity_boost": 0.0,
        },
        "synthesis": {
            "para_boost": 0.05, "sent_boost": 0.0, "deriv_boost": 0.10, "diversity_boost": 0.3,
        },
    }

    def retrieve_adaptive(
        self,
        query: str,
        k: int = 10,
        query_type: str | None = None,
        profile: DocumentProfile | None = None,
        focus_doc: str | None = None,
        metadata_filters: dict | None = None,
        metadata_boost: dict | None = None,
        use_rrf: bool = False,
    ) -> tuple[dict[int, list[tuple[IndexEntry, float]]], str]:
        """Adaptive fractal retrieval: uses flat for specification, reranked for other types.

        This is the recommended retrieval method. It auto-classifies the query type
        and selects the optimal strategy:
        - Specification queries: flat doc-level (precise, no noise from sub-doc)
        - Summary/Logic/Synthesis: reranked fractal (doc-primary + sub-doc boosts)

        Args:
            use_rrf: When True, routes non-specification queries to retrieve_rrf()
                     instead of retrieve_reranked(). RRF uses rank-based fusion
                     which is more robust to score scale differences.
        """
        qtype = query_type or classify_query_type(query, profile)
        boosts = dict(self._TYPE_BOOSTS.get(qtype, self._TYPE_BOOSTS["synthesis"]))

        # Auto-detect domain hints if no explicit metadata_boost provided
        effective_metadata_boost = metadata_boost
        if effective_metadata_boost is None:
            hints = extract_domain_hints(query)
            if hints:
                effective_metadata_boost = hints

        # Extract diversity_boost from boosts (not a retrieve_reranked kwarg via **)
        diversity = boosts.pop("diversity_boost", 0.0)

        if boosts["sent_boost"] == 0 and boosts["para_boost"] == 0 and boosts["deriv_boost"] == 0:
            # Pure flat retrieval for this query type
            return self.retrieve(
                query, k=k, levels=[2],
                query_type=qtype, profile=profile, focus_doc=focus_doc,
                use_derivatives=False, use_level_weights=False,
                metadata_filters=metadata_filters,
            )
        elif use_rrf:
            return self.retrieve_rrf(
                query, k=k,
                use_derivatives=(boosts["deriv_boost"] > 0),
                metadata_filters=metadata_filters,
                metadata_boost=effective_metadata_boost,
            )
        else:
            return self.retrieve_reranked(
                query, k=k,
                query_type=qtype, profile=profile, focus_doc=focus_doc,
                use_derivatives=(boosts["deriv_boost"] > 0),
                metadata_filters=metadata_filters,
                metadata_boost=effective_metadata_boost,
                diversity_boost=diversity,
                **boosts,
            )

    def retrieve_rrf(
        self,
        query: str,
        k: int = 10,
        rrf_k: int = 60,
        levels: list[int] | None = None,
        use_derivatives: bool = True,
        metadata_filters: dict | None = None,
        metadata_boost: dict | None = None,
    ) -> tuple[dict[int, list[tuple[IndexEntry, float]]], str]:
        """Reciprocal Rank Fusion across fractal levels.

        Score = SUM over levels of 1/(rrf_k + rank_in_level)

        More robust than linear boost combination — uses rank position
        instead of raw cosine similarity scores.

        Args:
            rrf_k: RRF constant (default 60, from Cormack et al.).
        """
        qtype = classify_query_type(query)
        qvec = self._embed(query)

        if levels is None:
            levels = [2, 1, 0]

        # Get per-level rankings (best score per doc)
        level_rankings: dict[int, list[tuple[str, float]]] = {}
        # Track best entries per doc per level for result building
        level_best_entries: dict[int, dict[str, tuple[IndexEntry, float]]] = {}

        for lvl in levels:
            candidates = self.index[lvl]
            if metadata_filters:
                candidates = [c for c in candidates
                              if self._filter_by_metadata(self._get_doc_id(c), metadata_filters)]

            doc_best: dict[str, float] = {}
            doc_best_entry: dict[str, tuple[IndexEntry, float]] = {}
            for item in candidates:
                doc_id = self._get_doc_id(item)
                sim = float(np.dot(qvec, item.vec))

                # Add derivative bonus for L0 entries
                if use_derivatives and item.id in self.derivatives:
                    d = self.derivatives[item.id]
                    sim += (float(np.dot(qvec, d["d1"])) * BASE_DERIV_WEIGHT +
                            float(np.dot(qvec, d["d2"])) * BASE_DERIV_WEIGHT * 0.6)

                if doc_id not in doc_best or sim > doc_best[doc_id]:
                    doc_best[doc_id] = sim
                    doc_best_entry[doc_id] = (item, sim)

            ranked = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)
            level_rankings[lvl] = ranked
            level_best_entries[lvl] = doc_best_entry

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        for _lvl, ranked in level_rankings.items():
            for rank, (doc_id, _) in enumerate(ranked, 1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # Add metadata boost
        if metadata_boost:
            for doc_id in rrf_scores:
                rrf_scores[doc_id] += self._compute_metadata_boost(doc_id, metadata_boost)

        # Sort and build results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results: dict[int, list[tuple[IndexEntry, float]]] = {lvl: [] for lvl in levels}
        for doc_id, rrf_score in sorted_docs:
            for lvl in levels:
                if doc_id in level_best_entries[lvl]:
                    entry, sim = level_best_entries[lvl][doc_id]
                    results[lvl].append((entry, rrf_score))

        return results, qtype

    def retrieve_flat(
        self,
        query: str,
        k: int = 3,
        metadata_filters: dict | None = None,
    ) -> tuple[dict[int, list[tuple[IndexEntry, float]]], str]:
        """Flat document-level-only retrieval (baseline for benchmarking)."""
        return self.retrieve(
            query, k=k, levels=[2],
            use_derivatives=False, use_level_weights=False,
            metadata_filters=metadata_filters,
        )

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return {
            "documents": len(self.docs),
            "entries_level_0": len(self.index[0]),
            "entries_level_1": len(self.index[1]),
            "entries_level_2": len(self.index[2]),
            "derivatives": len(self.derivatives),
            "adapters": len(self.adapters),
            "documents_with_metadata": len(self.doc_metadata),
            "embedding_dim": self.dim,
            "backend": type(self._backend).__name__,
        }
