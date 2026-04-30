"""Pipeline executor that runs composed technique sequences.

The executor takes an ``ExecutionPlan`` (from the registry/composer) and
a ``PipelineContext``, then runs each technique step in order.  Built-in
steps cover the default retrieval pipeline; new techniques are registered
at startup and the executor discovers them by name.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from cam_rag.generation.prompt import build_rag_prompt
from cam_rag.pipeline.context import PipelineContext
from cam_rag.pipeline.governance import FitnessScore, FitnessTracker, LifecycleManager
from cam_rag.pipeline.registry import ExecutionPlan
from cam_rag.rag.models import Citation, Evidence
from cam_rag.reranking.prompt import build_rerank_prompt, parse_rerank_scores
from cam_rag.pipeline.query_decompose import QueryDecomposeStep
from cam_rag.pipeline.reciprocal_neighbor import ReciprocalNeighborStep
from cam_rag.pipeline.score_normalize import ScoreNormalizeStep
from cam_rag.pipeline.title_boost import TitleBoostStep
from cam_rag.retrieval.adaptive_fusion import compute_idf_adaptive_weights
from cam_rag.retrieval.dense import DenseVectorRetriever
from cam_rag.retrieval.hyde import HyDEStep
from cam_rag.retrieval.fusion import rrf_fuse
from cam_rag.retrieval.query_expansion import build_expanded_query, extract_expansion_terms
from cam_rag.retrieval.sparse import SparseBM25Retriever
from cam_rag.pipeline.routing import ComplexityRoutingStep
from cam_rag.scoring.contracts import apply_selective_filter, classify_contracts
from cam_rag.scoring.moe import score_chunks_moe
from cam_rag.verification.confidence import score_retrieval_confidence
from cam_rag.verification.etf import compute_etf
from cam_rag.verification.grounding import verify_citations_grounded

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TechniqueStep protocol
# ---------------------------------------------------------------------------


class TechniqueStep(Protocol):
    """Contract that every executable technique step must satisfy."""

    @property
    def name(self) -> str: ...

    def execute(self, ctx: PipelineContext) -> None:
        """Mutate *ctx* in place with this step's output."""
        ...


# ---------------------------------------------------------------------------
# Built-in technique step implementations
# ---------------------------------------------------------------------------


class SparseBM25Step:
    """Build the sparse index and run BM25 retrieval."""

    name = "sparse_bm25"

    def execute(self, ctx: PipelineContext) -> None:
        if ctx.sparse_retriever is None:
            ctx.sparse_retriever = SparseBM25Retriever(
                ctx.retrieval_docs, tokenizer=ctx.spec.tokenize,
            )
        sparse_k = max(ctx.limit, ctx.adaptive.sparse_k)
        ctx.sparse_results = ctx.sparse_retriever.retrieve(ctx.expanded_query, k=sparse_k)


class DenseVectorStep:
    """Build the dense index and run vector retrieval.

    When ``ctx.spec.embedding_backends`` is non-empty, runs one
    ``DenseVectorRetriever`` per backend and stores all result sets in
    ``ctx.extras["ensemble_dense_results"]`` for downstream
    ``RRFFusionStep`` to consume.  When the list is empty, falls back to
    the single-backend path for full backward compatibility.
    """

    name = "dense_vector"

    def execute(self, ctx: PipelineContext) -> None:
        backends = ctx.spec.embedding_backends

        if not backends:
            # --- Single-backend path (backward compatible) ---
            if ctx.dense_retriever is None:
                ctx.dense_retriever = DenseVectorRetriever(
                    ctx.retrieval_docs,
                    backend=ctx.spec.embedding_backend,
                )
            dense_k = max(ctx.limit, ctx.adaptive.dense_k)
            ctx.dense_results = ctx.dense_retriever.retrieve(
                ctx.expanded_query, k=dense_k,
            )
            return

        # --- Ensemble path: multiple dense backends ---
        dense_k = max(ctx.limit, ctx.adaptive.dense_k)
        ensemble_results: dict[str, list] = {}
        ensemble_retrievers: dict[str, DenseVectorRetriever] = {}

        # Reuse previously built retrievers if they exist
        prev_retrievers = ctx.extras.get("ensemble_dense_retrievers", {})

        for i, backend in enumerate(backends):
            name = f"dense_{i}"
            if name in prev_retrievers:
                retriever = prev_retrievers[name]
            else:
                retriever = DenseVectorRetriever(
                    ctx.retrieval_docs, backend=backend,
                )
            ensemble_retrievers[name] = retriever
            results = retriever.retrieve(ctx.expanded_query, k=dense_k)
            ensemble_results[name] = results

        # Store ensemble results and retrievers for RRFFusionStep
        # and QueryExpansionStep
        ctx.extras["ensemble_dense_results"] = ensemble_results
        ctx.extras["ensemble_dense_retrievers"] = ensemble_retrievers

        # Also set ctx.dense_results to the first backend's results
        # for backward compatibility with any code that reads it
        first_key = next(iter(ensemble_results))
        ctx.dense_results = ensemble_results[first_key]


class AdaptiveFusionStep:
    """Compute IDF-adaptive fusion weights from the query and BM25 index.

    When ``ctx.spec.adaptive_fusion_enabled`` is True and a sparse retriever
    is available, this step analyses query-term IDF to shift sparse/dense
    fusion weights before RRF runs.  No-op otherwise.
    """

    name = "adaptive_fusion_weights"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.adaptive_fusion_enabled:
            return
        if ctx.sparse_retriever is None:
            return

        query_terms = ctx.spec.tokenize(ctx.expanded_query)
        if not query_terms:
            return

        # Extract document frequencies from the BM25 retriever
        df = getattr(ctx.sparse_retriever, "_document_frequencies", None)
        if df is None:
            return

        num_docs = len(ctx.sparse_retriever.documents)
        dense_w, sparse_w = compute_idf_adaptive_weights(
            query_terms,
            df,
            num_docs,
            base_dense_weight=ctx.adaptive.dense_weight,
            base_sparse_weight=ctx.adaptive.sparse_weight,
        )
        ctx.adaptive.dense_weight = dense_w
        ctx.adaptive.sparse_weight = sparse_w


class RRFFusionStep:
    """Fuse sparse + dense results with Reciprocal Rank Fusion.

    When ``ctx.extras["ensemble_dense_results"]`` is present, fuses sparse
    plus all dense backend result lists (N+1 way fusion).  Otherwise falls
    back to the standard 2-way sparse+dense fusion.
    """

    name = "rrf_fusion"

    def execute(self, ctx: PipelineContext) -> None:
        ensemble_results = ctx.extras.get("ensemble_dense_results")

        if ensemble_results:
            # Ensemble mode: fuse sparse + all dense backends
            ranked_lists = [ctx.sparse_results]
            names = ["sparse"]

            # Build weights dict
            weights: dict[str, float] = {"sparse": ctx.adaptive.sparse_weight}

            # Distribute the dense_weight across ensemble backends
            num_dense = len(ensemble_results)
            spec_weights = ctx.spec.ensemble_weights

            for dense_name, dense_result_list in ensemble_results.items():
                ranked_lists.append(dense_result_list)
                names.append(dense_name)
                if dense_name in spec_weights:
                    weights[dense_name] = spec_weights[dense_name]
                else:
                    # Equal share of dense_weight
                    weights[dense_name] = ctx.adaptive.dense_weight / num_dense

            # Include retriever plugin results in N-way fusion
            _add_plugin_results(ctx, ranked_lists, names, weights)

            ctx.fused_results = rrf_fuse(
                *ranked_lists,
                names=names,
                rrf_k=ctx.adaptive.rrf_k,
                weights=weights,
            )[:ctx.limit]
        else:
            ranked_lists = [ctx.dense_results, ctx.sparse_results]
            names = ["dense", "sparse"]
            weights = {
                "dense": ctx.adaptive.dense_weight,
                "sparse": ctx.adaptive.sparse_weight,
            }

            # Include retriever plugin results in N-way fusion
            _add_plugin_results(ctx, ranked_lists, names, weights)

            ctx.fused_results = rrf_fuse(
                *ranked_lists,
                names=names,
                rrf_k=ctx.adaptive.rrf_k,
                weights=weights,
            )[:ctx.limit]

        ctx.evidence = _fused_to_evidence(ctx)


class QueryExpansionStep:
    """Expand the query from top evidence and re-run retrieval."""

    name = "query_expansion"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.query_expansion_enabled or not ctx.evidence:
            return

        expanded = build_expanded_query(
            ctx.query_text,
            ctx.evidence,
            tokenizer=ctx.spec.tokenize,
            max_terms=ctx.spec.expansion_terms,
        )
        if expanded == ctx.query_text:
            return

        ctx.expanded_query = expanded

        # Re-run sparse retrieval with expanded query
        sparse_k = max(ctx.limit, ctx.adaptive.sparse_k)
        dense_k = max(ctx.limit, ctx.adaptive.dense_k)

        if ctx.sparse_retriever is not None:
            ctx.sparse_results = ctx.sparse_retriever.retrieve(expanded, k=sparse_k)

        # Re-run dense retrieval (ensemble or single)
        ensemble_retrievers = ctx.extras.get("ensemble_dense_retrievers")
        if ensemble_retrievers:
            # Ensemble re-run: reuse the same retriever instances
            ensemble_results: dict[str, list] = {}
            for name, retriever in ensemble_retrievers.items():
                results = retriever.retrieve(expanded, k=dense_k)
                ensemble_results[name] = results

            ctx.extras["ensemble_dense_results"] = ensemble_results

            first_key = next(iter(ensemble_results))
            ctx.dense_results = ensemble_results[first_key]
        elif ctx.dense_retriever is not None:
            ctx.dense_results = ctx.dense_retriever.retrieve(expanded, k=dense_k)

        # Re-fuse with the appropriate mode
        ensemble_results_final = ctx.extras.get("ensemble_dense_results")
        if ensemble_results_final:
            ranked_lists = [ctx.sparse_results]
            names = ["sparse"]
            weights: dict[str, float] = {"sparse": ctx.adaptive.sparse_weight}

            num_dense = len(ensemble_results_final)
            spec_weights = ctx.spec.ensemble_weights

            for dense_name, dense_result_list in ensemble_results_final.items():
                ranked_lists.append(dense_result_list)
                names.append(dense_name)
                if dense_name in spec_weights:
                    weights[dense_name] = spec_weights[dense_name]
                else:
                    weights[dense_name] = ctx.adaptive.dense_weight / num_dense

            ctx.fused_results = rrf_fuse(
                *ranked_lists,
                names=names,
                rrf_k=ctx.adaptive.rrf_k,
                weights=weights,
            )[:ctx.limit]
        else:
            ctx.fused_results = rrf_fuse(
                ctx.dense_results,
                ctx.sparse_results,
                names=["dense", "sparse"],
                rrf_k=ctx.adaptive.rrf_k,
                weights={
                    "dense": ctx.adaptive.dense_weight,
                    "sparse": ctx.adaptive.sparse_weight,
                },
            )[:ctx.limit]

        ctx.evidence = _fused_to_evidence(ctx)


class MultiHopRetrievalStep:
    """Iteratively refine retrieval by extracting concepts from evidence and re-querying.

    For each hop beyond the first:
    1. Extract key terms from current evidence that are not in the original query
    2. Build a follow-up query combining the original query + new terms
    3. Re-run sparse retrieval with the follow-up query
    4. Merge new results with existing evidence (deduplicate by chunk ID)
    5. Re-score and re-sort

    Each new evidence item receives a ``"hop"`` signal indicating which
    iteration discovered it.  The total evidence count is capped at
    ``ctx.limit`` after merging.
    """

    name = "multi_hop_retrieval"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.multi_hop_enabled:
            return
        if not ctx.evidence:
            return

        max_hops = ctx.spec.multi_hop_max_hops

        # Tag existing evidence as hop 1
        for item in ctx.evidence:
            if "hop" not in item.signals:
                item.signals["hop"] = 1

        existing_chunk_ids = {item.chunk.id for item in ctx.evidence}
        original_query_terms = set(ctx.spec.tokenize(ctx.query_text))

        for hop_number in range(2, max_hops + 1):
            # Filter out "fast" items for term extraction (they are low-value)
            extraction_evidence = [
                item for item in ctx.evidence
                if item.signals.get("routing") != "fast"
            ] or ctx.evidence  # fallback to all if everything is fast

            # Extract new terms from current evidence that are not in the original query
            new_terms = extract_expansion_terms(
                extraction_evidence,
                ctx.query_text,
                tokenizer=ctx.spec.tokenize,
                top_k=min(5, len(extraction_evidence)),
                max_terms=ctx.spec.expansion_terms,
            )
            if not new_terms:
                logger.debug(
                    "multi-hop: no new terms extracted at hop %d, stopping",
                    hop_number,
                )
                break

            # Build the follow-up query
            follow_up_query = ctx.query_text + " " + " ".join(new_terms)

            # Re-run sparse retrieval with the follow-up query
            if ctx.sparse_retriever is None:
                break

            sparse_k = max(ctx.limit, ctx.adaptive.sparse_k)
            hop_sparse_results = ctx.sparse_retriever.retrieve(
                follow_up_query, k=sparse_k,
            )

            # Convert sparse results to evidence and merge (deduplicate by chunk ID)
            hop_query_terms = set(ctx.spec.tokenize(follow_up_query))
            new_evidence: list[Evidence] = []
            for result in hop_sparse_results:
                if result.doc_id in existing_chunk_ids:
                    continue
                chunk = ctx.chunk_by_id.get(result.doc_id)
                if chunk is None:
                    continue
                matched_terms = sorted(
                    hop_query_terms.intersection(ctx.spec.tokenize(chunk.text))
                )
                new_evidence.append(
                    Evidence(
                        chunk=chunk,
                        score=result.score,
                        retriever="multi_hop_sparse",
                        rank=0,
                        signals={
                            "hop": hop_number,
                            "matched_terms": matched_terms,
                            "hop_query": follow_up_query,
                        },
                    )
                )
                existing_chunk_ids.add(result.doc_id)

            if not new_evidence:
                logger.debug(
                    "multi-hop: no new evidence at hop %d, stopping",
                    hop_number,
                )
                break

            # Merge new evidence with existing, then re-sort and trim
            ctx.evidence.extend(new_evidence)
            ctx.evidence.sort(key=lambda e: e.score, reverse=True)
            ctx.evidence = ctx.evidence[:ctx.limit]

            # Update ranks after sort
            for i, item in enumerate(ctx.evidence):
                item.rank = i


class ConfidenceScoringStep:
    """Score retrieval confidence from evidence signals."""

    name = "confidence_scoring"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.evidence:
            ctx.confidence_score = 0.0
            return
        # Build citations from evidence (needed for grounding step)
        ctx.citations = [
            Citation(
                source=item.chunk.source,
                document_id=item.chunk.document_id,
                title=item.chunk.title,
                section_heading=item.chunk.section_heading,
                excerpt=item.chunk.text[:240],
                score=item.score,
            )
            for item in ctx.evidence
        ]
        report = score_retrieval_confidence(ctx.evidence)
        ctx.confidence_score = report.overall
        ctx.confidence_details = report.to_dict()


class CitationGroundingStep:
    """Verify that all citations are grounded in retrieved evidence."""

    name = "citation_grounding"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.citations or not ctx.evidence:
            ctx.grounded = True  # vacuously true
            return
        report = verify_citations_grounded(ctx.citations, ctx.evidence)
        ctx.grounded = report.grounded
        ctx.confidence_details["grounding"] = report.to_dict()


class DenseToEvidenceStep:
    """Convert dense retrieval results directly to Evidence, bypassing RRF.

    Used by the ``dense_dominant`` strategy when BM25/RRF would dilute
    strong dense embeddings.  The dense retriever's cosine similarity
    scores become the evidence scores directly.
    """

    name = "dense_to_evidence"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.dense_results:
            return
        query_terms = set(ctx.spec.tokenize(ctx.expanded_query))
        evidence: list[Evidence] = []
        for i, result in enumerate(ctx.dense_results[:ctx.limit]):
            chunk = ctx.chunk_by_id.get(result.doc_id)
            if chunk is None:
                continue
            matched_terms = sorted(
                query_terms.intersection(ctx.spec.tokenize(chunk.text))
            )
            evidence.append(
                Evidence(
                    chunk=chunk,
                    score=result.score,
                    retriever="dense_only",
                    rank=i,
                    signals={
                        "matched_terms": matched_terms,
                        "strategy": "dense_dominant",
                    },
                )
            )
        ctx.evidence = evidence


class CrossEncoderRerankStep:
    """Re-score evidence using a cross-encoder reranking backend.

    Requires ``ctx.spec.reranker_backend`` to be set.  When no backend is
    configured, this step is a no-op.  When enabled, each evidence item's
    score is replaced by the reranker's relevance score and the list is
    re-sorted by descending score.  The original RRF score is preserved
    in ``signals["rrf_score_pre_rerank"]``.
    """

    name = "cross_encoder_rerank"

    def execute(self, ctx: PipelineContext) -> None:
        backend = ctx.spec.reranker_backend
        if backend is None:
            return
        if not ctx.evidence:
            return

        # Separate items to rerank from "fast" items that skip reranking
        to_rerank = []
        fast_items = []
        for item in ctx.evidence:
            if item.signals.get("routing") == "fast":
                fast_items.append(item)
            else:
                to_rerank.append(item)

        if to_rerank:
            passages = [item.chunk.text for item in to_rerank]
            scores = backend.rerank(ctx.query_text, passages)

            for item, rerank_score in zip(to_rerank, scores):
                item.signals["rrf_score_pre_rerank"] = item.score
                item.signals["reranker_score"] = rerank_score
                item.score = rerank_score

        # Merge back: reranked items + fast items (fast keep original scores)
        all_items = to_rerank + fast_items

        # Re-sort by score (descending) and update ranks
        all_items.sort(key=lambda e: e.score, reverse=True)
        for i, item in enumerate(all_items):
            item.rank = i

        ctx.evidence = all_items


class GenerationStep:
    """Synthesize an LLM answer from retrieved evidence.

    Requires ``ctx.spec.generation_backend`` to be set.  When no backend
    is configured or no evidence is available, falls back to the
    retrieval-only template answer.
    """

    name = "generation"

    def execute(self, ctx: PipelineContext) -> None:
        backend = ctx.spec.generation_backend
        if backend is None:
            return
        if not ctx.evidence:
            ctx.generated_answer = (
                f"No evidence was retrieved for: {ctx.query_text}"
            )
            return

        system_prompt, user_prompt = build_rag_prompt(
            ctx.query_text, ctx.evidence,
            contract_aware=ctx.spec.accuracy_contracts_enabled,
        )
        generated = backend.generate(system_prompt, user_prompt)
        if generated:
            ctx.generated_answer = generated
        else:
            logger.warning(
                "Generation backend returned empty; falling back to "
                "retrieval-only answer"
            )


class MoEImportanceScoringStep:
    """Score chunks using the 7-expert MoE panel.

    Enriches ``chunk.metadata`` with importance scores.
    No-op when ``ctx.spec.moe_scoring_enabled`` is False.
    """

    name = "moe_importance_scoring"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.moe_scoring_enabled:
            return
        if not ctx.chunks:
            return
        score_chunks_moe(ctx.chunks)


class AccuracyContractStep:
    """Classify chunks into hard/medium/soft accuracy contract tiers.

    Requires MoE scoring to have run first (reads ``moe_votes`` from
    chunk metadata).  No-op when ``ctx.spec.accuracy_contracts_enabled``
    is False.
    """

    name = "accuracy_contracts"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.accuracy_contracts_enabled:
            return
        if not ctx.chunks:
            return
        classify_contracts(ctx.chunks)


class SelectiveFilterStep:
    """Adjust evidence scores by MoE store-score worthiness.

    Boosts high-value evidence and penalises boilerplate.
    No-op when ``ctx.spec.selective_filter_enabled`` is False
    or no evidence is available.
    """

    name = "selective_filter"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.selective_filter_enabled:
            return
        if not ctx.evidence:
            return
        apply_selective_filter(
            ctx.evidence,
            threshold=ctx.spec.selective_filter_threshold,
        )


class ETFVerificationStep:
    """Check answer vs evidence grounding using Epistemic Tension Field.

    Evaluates context sovereignty (token overlap) and model prior skeptic
    (generic phrase detection).  No-op when
    ``ctx.spec.etf_verification_enabled`` is False or no generated answer
    is available.
    """

    name = "etf_verification"

    def execute(self, ctx: PipelineContext) -> None:
        if not ctx.spec.etf_verification_enabled:
            return
        if not ctx.generated_answer:
            return
        report = compute_etf(ctx.generated_answer, ctx.evidence)
        ctx.extras["etf_report"] = report.to_dict()
        ctx.confidence_details["etf"] = report.to_dict()


# ---------------------------------------------------------------------------
# Step registry (name -> instance)
# ---------------------------------------------------------------------------

_BUILTIN_STEPS: dict[str, TechniqueStep] = {}


def _register_builtins() -> None:
    for step_cls in (
        SparseBM25Step,
        HyDEStep,
        DenseVectorStep,
        AdaptiveFusionStep,
        RRFFusionStep,
        DenseToEvidenceStep,
        TitleBoostStep,
        MoEImportanceScoringStep,
        AccuracyContractStep,
        SelectiveFilterStep,
        ComplexityRoutingStep,
        QueryExpansionStep,
        QueryDecomposeStep,
        CrossEncoderRerankStep,
        ReciprocalNeighborStep,
        MultiHopRetrievalStep,
        ScoreNormalizeStep,
        ConfidenceScoringStep,
        CitationGroundingStep,
        GenerationStep,
        ETFVerificationStep,
    ):
        instance = step_cls()
        _BUILTIN_STEPS[instance.name] = instance


_register_builtins()


def register_step(step: TechniqueStep) -> None:
    """Register a custom technique step by name."""
    _BUILTIN_STEPS[step.name] = step


def get_step(name: str) -> TechniqueStep:
    """Look up a step by name.  Raises ``KeyError`` if not found."""
    if name not in _BUILTIN_STEPS:
        raise KeyError(f"No technique step registered for: {name}")
    return _BUILTIN_STEPS[name]


# ---------------------------------------------------------------------------
# Pipeline executor
# ---------------------------------------------------------------------------


class PipelineExecutor:
    """Execute a composed ``ExecutionPlan`` against a ``PipelineContext``."""

    def run(
        self,
        plan: ExecutionPlan,
        ctx: PipelineContext,
        *,
        fitness_tracker: FitnessTracker | None = None,
        lifecycle_manager: LifecycleManager | None = None,
    ) -> PipelineContext:
        """Run every step in *plan* sequentially, mutating *ctx* in place.

        When *fitness_tracker* is provided, each step is timed and a
        ``FitnessScore`` observation is recorded after all steps complete.
        When *lifecycle_manager* is also provided, lifecycle state
        transitions are evaluated at the end of the run.
        """
        for step_name in plan.steps:
            step = get_step(step_name)
            t0 = time.perf_counter()
            step.execute(ctx)
            elapsed = time.perf_counter() - t0
            ctx.steps_executed.append(step_name)
            ctx.step_timings[step_name] = elapsed

        if fitness_tracker is not None:
            self._record_fitness(plan, ctx, fitness_tracker)

        if lifecycle_manager is not None and fitness_tracker is not None:
            lifecycle_manager.evaluate_transitions(fitness_tracker)

        return ctx

    def _record_fitness(
        self,
        plan: ExecutionPlan,
        ctx: PipelineContext,
        tracker: FitnessTracker,
    ) -> None:
        """Record a FitnessScore observation for each step in the plan."""
        evidence_ratio = (
            min(1.0, len(ctx.evidence) / ctx.limit)
            if ctx.limit > 0
            else 0.0
        )
        for step_name in plan.steps:
            timing = ctx.step_timings.get(step_name, 0.0)
            latency_score = max(0.0, 1.0 - timing / 5.0)
            score = FitnessScore(
                retrieval_quality=evidence_ratio,
                query_type_affinity=0.5,
                confidence_lift=0.5,
                diversity_contribution=0.5,
                latency_efficiency=latency_score,
                recency=1.0,
            )
            tracker.record(step_name, score)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_plugin_results(
    ctx: PipelineContext,
    ranked_lists: list,
    names: list[str],
    weights: dict[str, float],
) -> None:
    """Add retriever plugin results to the fusion inputs.

    Each plugin's results are added as a separate ranked list with
    equal weight share (distributed evenly across all plugins).
    """
    plugins = getattr(ctx.spec, "retriever_plugins", None)
    if not plugins:
        return

    # Index plugins if not already done
    plugin_indexed = ctx.extras.get("_plugins_indexed", set())
    for plugin in plugins:
        if plugin.name not in plugin_indexed:
            try:
                plugin.index(ctx.retrieval_docs)
                plugin_indexed.add(plugin.name)
            except Exception:
                logger.warning(
                    "Plugin %s failed to index, skipping", plugin.name,
                )
                continue
    ctx.extras["_plugins_indexed"] = plugin_indexed

    # Retrieve from each plugin
    k = max(ctx.limit, ctx.adaptive.dense_k)
    # Give plugins an equal share of weight (0.1 per plugin by default)
    plugin_weight = 0.1
    for plugin in plugins:
        try:
            results = plugin.retrieve(ctx.expanded_query, k=k)
        except Exception:
            logger.warning(
                "Plugin %s failed to retrieve, skipping", plugin.name,
            )
            continue
        if results:
            ranked_lists.append(results)
            names.append(f"plugin_{plugin.name}")
            weights[f"plugin_{plugin.name}"] = plugin_weight


def _fused_to_evidence(ctx: PipelineContext) -> list[Evidence]:
    """Convert fused results back to Evidence using the chunk map."""
    query_terms = set(ctx.spec.tokenize(ctx.expanded_query))
    evidence: list[Evidence] = []
    for result in ctx.fused_results:
        chunk = ctx.chunk_by_id.get(result.doc_id)
        if chunk is None:
            continue
        matched_terms = sorted(query_terms.intersection(ctx.spec.tokenize(chunk.text)))
        evidence.append(
            Evidence(
                chunk=chunk,
                score=result.rrf_score,
                retriever="hybrid_rrf",
                rank=result.rank,
                signals={
                    "matched_terms": matched_terms,
                    "source_ranks": result.source_ranks,
                    "source_scores": result.source_scores,
                    "rrf_score": result.rrf_score,
                },
            )
        )
    return evidence
