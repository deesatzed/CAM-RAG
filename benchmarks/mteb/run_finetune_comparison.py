#!/usr/bin/env python3
"""Before/after MTEB benchmark comparison for fine-tuned embedding models.

Usage
-----
    python benchmarks/mteb/run_finetune_comparison.py \\
        --base-model local:all-MiniLM-L6-v2 \\
        --finetuned-model local:./output/finetuned \\
        --tasks SciFact NFCorpus

Both ``--base-model`` and ``--finetuned-model`` accept any model prefix
understood by ``cam_rag.benchmarks.mteb._load_model``:

- ``local:<path>`` — local sentence-transformers model
- ``openrouter:<model>`` — OpenRouter embedding API
- ``ollama:<model>`` — Ollama embedding API
- ``hash`` — deterministic hash baseline

Results are written to ``benchmarks/mteb/results/finetune_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from cam_rag.benchmarks.mteb import (
    CamRAGMTEBModel,
    _load_model,
    _load_tasks,
    build_arg_parser,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MTEB retrieval benchmarks before and after fine-tuning."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base (pre-fine-tuning) model specifier, e.g. local:all-MiniLM-L6-v2.",
    )
    parser.add_argument(
        "--finetuned-model",
        required=True,
        help="Fine-tuned model specifier, e.g. local:./output/finetuned.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["SciFact", "NFCorpus"],
        help="MTEB task names to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Explicit embedding dimension (for OpenRouter/Ollama backends).",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/mteb/results",
        help="Directory for comparison results.",
    )
    return parser


def _run_mteb(model: Any, tasks: Any, batch_size: int) -> dict[str, float]:
    """Run MTEB evaluation and extract per-task nDCG@10 scores."""
    import importlib

    mteb = importlib.import_module("mteb")

    model_result = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={"batch_size": batch_size},
        overwrite_strategy="always",
    )

    scores: dict[str, float] = {}
    task_results = getattr(model_result, "task_results", None)
    if task_results:
        for tr in task_results:
            name = getattr(tr, "task_name", None) or str(tr)
            main_score = getattr(tr, "main_score", None)
            if main_score is not None:
                scores[name] = round(float(main_score), 4)
    return scores


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import importlib

    mteb = importlib.import_module("mteb")
    tasks = mteb.get_tasks(tasks=args.tasks)

    # Build model args namespace compatible with _load_model
    base_parser = build_arg_parser()

    base_args = base_parser.parse_args([
        "--model", args.base_model,
        *(["--embedding-dim", str(args.embedding_dim)] if args.embedding_dim else []),
    ])
    ft_args = base_parser.parse_args([
        "--model", args.finetuned_model,
        *(["--embedding-dim", str(args.embedding_dim)] if args.embedding_dim else []),
    ])

    print(f"Evaluating base model: {args.base_model}")
    base_model = _load_model(base_args, mteb)
    base_scores = _run_mteb(base_model, tasks, args.batch_size)
    print(f"  Scores: {base_scores}")

    print(f"\nEvaluating fine-tuned model: {args.finetuned_model}")
    ft_model = _load_model(ft_args, mteb)
    ft_scores = _run_mteb(ft_model, tasks, args.batch_size)
    print(f"  Scores: {ft_scores}")

    # Compute deltas
    deltas: dict[str, float] = {}
    for task in set(base_scores) | set(ft_scores):
        b = base_scores.get(task, 0.0)
        f = ft_scores.get(task, 0.0)
        deltas[task] = round(f - b, 4)

    comparison = {
        "base_model": args.base_model,
        "finetuned_model": args.finetuned_model,
        "tasks": args.tasks,
        "base_scores": base_scores,
        "finetuned_scores": ft_scores,
        "deltas": deltas,
    }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / "finetune_comparison.json"
    result_file.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nComparison:")
    for task in sorted(deltas):
        b = base_scores.get(task, 0.0)
        f = ft_scores.get(task, 0.0)
        d = deltas[task]
        sign = "+" if d >= 0 else ""
        print(f"  {task}: {b:.4f} -> {f:.4f} ({sign}{d:.4f})")

    print(f"\nResults saved to {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
