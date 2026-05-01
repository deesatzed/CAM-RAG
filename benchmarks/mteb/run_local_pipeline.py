#!/usr/bin/env python3
"""Run full-pipeline MTEB benchmark with local models only (no API keys needed).

Uses all-MiniLM-L6-v2 for embeddings and cross-encoder/ms-marco-MiniLM-L-6-v2
for reranking. Both download automatically from HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local pipeline MTEB benchmark")
    parser.add_argument("--tasks", nargs="*", default=["SciFact", "NFCorpus"])
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--adaptive-fusion", action="store_true")
    parser.add_argument("--instruction-prefix", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-depth", type=int, default=100,
                        help="How many docs to retrieve before reranking")
    parser.add_argument("--dense-weight", type=float, default=0.6,
                        help="Dense retrieval weight for RRF fusion (0.0-1.0)")
    parser.add_argument("--sparse-weight", type=float, default=0.4,
                        help="Sparse BM25 weight for RRF fusion (0.0-1.0)")
    parser.add_argument("--pipeline-strategy", default="auto",
                        choices=["auto", "dense_only", "dense_dominant", "strong_hybrid", "hybrid", "sparse_boost", "blended_rerank"],
                        help="Pipeline strategy (auto detects embedding quality)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True to cross-encoder (needed for gemma models)")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Enable unsupervised auto-calibration of pipeline params")
    parser.add_argument("--output-dir", default="benchmarks/mteb/results/local-pipeline")
    args = parser.parse_args(argv)

    import importlib
    mteb = importlib.import_module("mteb")

    from cam_rag.benchmarks.search_protocol import CamRAGSearchModel
    from cam_rag.finetuning.backend import LocalSentenceTransformerBackend
    from cam_rag.rag.spec import RAGAppSpec

    # Build embedding backend
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_backend = LocalSentenceTransformerBackend(args.embedding_model)
    print(f"  dim={embedding_backend.dim}")

    # Optionally wrap with instruction prefix
    if args.instruction_prefix:
        from cam_rag.retrieval.instruction_embeddings import InstructionEmbeddingBackend
        embedding_backend = InstructionEmbeddingBackend(
            embedding_backend, instruction=args.instruction_prefix,
        )
        print(f"  instruction prefix: {args.instruction_prefix[:60]}...")

    # Build reranker
    reranker = None
    if not args.no_reranker:
        from cam_rag.reranking.cross_encoder import LocalCrossEncoderBackend
        print(f"Loading cross-encoder: {args.cross_encoder}")
        reranker = LocalCrossEncoderBackend(
            args.cross_encoder,
            trust_remote_code=args.trust_remote_code,
        )

    # Build spec
    spec = RAGAppSpec(
        name="local-pipeline-bench",
        use_pipeline=True,
        embedding_backend=embedding_backend,
        reranker_backend=reranker,
        adaptive_fusion_enabled=args.adaptive_fusion,
        retrieval_top_k=args.top_k,
        pipeline_strategy=args.pipeline_strategy,
        auto_calibrate=args.auto_calibrate,
    )

    model = CamRAGSearchModel(
        spec,
        name=f"pipeline-{args.embedding_model}",
        retrieval_depth=args.retrieval_depth,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    # Run MTEB
    tasks = mteb.get_tasks(tasks=args.tasks)
    print(f"\nRunning MTEB on tasks: {args.tasks}")
    t0 = time.time()

    model_result = mteb.evaluate(
        model,
        tasks=tasks,
        overwrite_strategy="always",
    )

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    # Extract scores
    scores = {}
    task_results = getattr(model_result, "task_results", None)
    if task_results:
        for tr in task_results:
            name = getattr(tr, "task_name", None) or str(tr)
            main_score = getattr(tr, "main_score", None)
            if main_score is not None:
                scores[name] = round(float(main_score), 5)

    print("\n=== Results ===")
    for task_name, score in sorted(scores.items()):
        print(f"  {task_name}: nDCG@10 = {score:.5f}")

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "embedding_model": args.embedding_model,
        "cross_encoder": args.cross_encoder if not args.no_reranker else None,
        "adaptive_fusion": args.adaptive_fusion,
        "instruction_prefix": args.instruction_prefix,
        "retrieval_depth": args.retrieval_depth,
        "auto_calibrate": args.auto_calibrate,
        "scores": scores,
        "elapsed_seconds": round(elapsed, 1),
    }
    result_file = output_path / "pipeline_scores.json"
    result_file.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved to {result_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
