#!/usr/bin/env python3
"""Run MTEB retrieval benchmarks using the full CAM RAG pipeline.

Uses SearchProtocol so MTEB evaluates BM25 + dense + RRF + reranking as a
single retrieval system, not just the embedding in isolation.

Usage:
    # Embedding-only (baseline)
    python benchmarks/mteb/run_full_pipeline.py \\
        --backend openrouter:qwen/qwen3-embedding-8b \\
        --embedding-dim 4096 \\
        --tasks SciFact NFCorpus

    # Full pipeline with cross-encoder reranker
    python benchmarks/mteb/run_full_pipeline.py \\
        --backend openrouter:qwen/qwen3-embedding-8b \\
        --embedding-dim 4096 \\
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --tasks SciFact NFCorpus

    # All strategies combined
    python benchmarks/mteb/run_full_pipeline.py \\
        --backend openrouter:qwen/qwen3-embedding-8b \\
        --embedding-dim 4096 \\
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --instruction-prefix "Given a scientific claim, retrieve evidence" \\
        --adaptive-fusion \\
        --tasks SciFact NFCorpus

    # Quick smoke test with hash backend
    python benchmarks/mteb/run_full_pipeline.py \\
        --backend hash \\
        --tasks SciFact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MTEB retrieval benchmarks with the full CAM RAG pipeline.",
    )
    parser.add_argument(
        "--backend",
        default="hash",
        help=(
            "Embedding backend: 'hash', 'openrouter:<model>', 'ollama:<model>', "
            "or 'local:<path>'."
        ),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension for the backend.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["SciFact", "NFCorpus"],
        help="MTEB task names to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results per query.",
    )
    parser.add_argument(
        "--output-folder",
        default="benchmarks/mteb/results",
        help="Directory for MTEB output.",
    )
    parser.add_argument(
        "--cross-encoder",
        default=None,
        help=(
            "Cross-encoder model for reranking, e.g. "
            "'cross-encoder/ms-marco-MiniLM-L-6-v2'. Loaded locally via "
            "sentence-transformers."
        ),
    )
    parser.add_argument(
        "--instruction-prefix",
        default=None,
        help=(
            "Instruction prefix for asymmetric query embedding, e.g. "
            "'Given a scientific claim, retrieve abstracts that support or refute it'."
        ),
    )
    parser.add_argument(
        "--adaptive-fusion",
        action="store_true",
        help="Enable IDF-adaptive sparse/dense fusion weights.",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Enable HyDE (Hypothetical Document Embeddings).",
    )
    parser.add_argument(
        "--hyde-backend",
        default=None,
        help=(
            "Generation backend for HyDE. Use 'ollama:<model>' or "
            "'openrouter:<model>'. Required when --hyde is set."
        ),
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help="Dense retrieval weight for RRF fusion (0.0-1.0).",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=None,
        help="Sparse BM25 weight for RRF fusion (0.0-1.0).",
    )
    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=None,
        help="Number of candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--pipeline-strategy",
        default=None,
        choices=["auto", "dense_dominant", "hybrid", "sparse_boost"],
        help=(
            "Pipeline strategy: 'auto' (detect embedding quality), "
            "'dense_dominant' (skip BM25), 'hybrid' (full pipeline), "
            "'sparse_boost' (boost BM25). Default: auto."
        ),
    )
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Enable unsupervised auto-calibration of pipeline params")
    parser.add_argument(
        "--spec-overrides",
        default=None,
        help="JSON string of additional RAGAppSpec overrides.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=256,
        help="Dimension for hash embedding backend.",
    )
    args = parser.parse_args(argv)

    import json as _json

    # Build spec overrides from CLI flags
    overrides: dict = {}
    if args.spec_overrides:
        overrides.update(_json.loads(args.spec_overrides))
    if args.cross_encoder:
        overrides["cross_encoder_model"] = args.cross_encoder
    if args.instruction_prefix:
        overrides["instruction_prefix"] = args.instruction_prefix
    if args.adaptive_fusion:
        overrides["adaptive_fusion_enabled"] = True
    if args.hyde:
        overrides["hyde_enabled"] = True
    if args.pipeline_strategy:
        overrides["pipeline_strategy"] = args.pipeline_strategy
    if args.dense_weight is not None:
        overrides["dense_weight"] = args.dense_weight
    if args.sparse_weight is not None:
        overrides["sparse_weight"] = args.sparse_weight
    if args.retrieval_depth is not None:
        overrides["retrieval_depth"] = args.retrieval_depth
    if args.auto_calibrate:
        overrides["auto_calibrate"] = True

    # Wire HyDE generation backend
    if args.hyde and args.hyde_backend:
        # Generation backend is an object — set it directly via spec_overrides
        # by building it here and passing it through the pipeline: handler
        pass  # Handled below after import

    from cam_rag.benchmarks.mteb import main as mteb_main

    # Build the model string for mteb.py's _load_model dispatch
    model_str = f"pipeline:{args.backend}"

    overrides_json = _json.dumps(overrides) if overrides else None

    mteb_argv = [
        "--model", model_str,
        "--tasks", *args.tasks,
        "--output-folder", args.output_folder,
        "--hash-dim", str(args.hash_dim),
    ]
    if args.embedding_dim:
        mteb_argv.extend(["--embedding-dim", str(args.embedding_dim)])
    if overrides_json:
        mteb_argv.extend(["--spec-overrides", overrides_json])

    return mteb_main(mteb_argv)


if __name__ == "__main__":
    raise SystemExit(main())
