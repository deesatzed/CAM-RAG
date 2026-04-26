"""MTEB benchmark runner for CAM RAG embedding backends."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cam_rag.retrieval import EmbeddingBackend, HashEmbeddingBackend


class CamRAGMTEBModel:
    """MTEB-compatible wrapper around a `cam_rag` embedding backend."""

    def __init__(self, backend: EmbeddingBackend, *, name: str = "cam-rag-hash") -> None:
        self.backend = backend
        self.mteb_model_meta = {"name": name, "revision": None, "release_date": None}

    def encode(self, inputs: Any, **_: Any) -> list[list[float]]:
        """Encode MTEB inputs using the wrapped backend.

        MTEB has supported multiple encode signatures over time. This wrapper
        accepts a plain list of strings as well as iterable batches from the
        current DataLoader-based protocol.
        """

        return [self.backend.embed(text) for text in _coerce_texts(inputs)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MTEB retrieval benchmarks for CAM RAG embedding backends."
    )
    parser.add_argument(
        "--model",
        default="hash",
        help=(
            "Model to evaluate. Use 'hash' for cam_rag.HashEmbeddingBackend, "
            "or any model accepted by mteb.get_model."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default="MTEB(eng, v2)",
        help="MTEB benchmark name used when --tasks is omitted.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Explicit MTEB task names. Defaults to retrieval tasks from --benchmark.",
    )
    parser.add_argument(
        "--task-type",
        default="Retrieval",
        help="Task type filter applied to --benchmark when --tasks is omitted.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional ISO 639-3 language filters passed to mteb.get_tasks.",
    )
    parser.add_argument(
        "--output-folder",
        default="benchmarks/mteb/results",
        help="Directory where MTEB writes results.",
    )
    parser.add_argument(
        "--prediction-folder",
        default=None,
        help="Optional directory for per-task ranking predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size forwarded to the model encode call.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=256,
        help="Embedding dimension for --model hash.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mteb = _load_mteb()
    model = _load_model(args, mteb)
    tasks = _load_tasks(args, mteb)

    evaluate_kwargs: dict[str, Any] = {
        "tasks": tasks,
        "output_folder": args.output_folder,
        "encode_kwargs": {"batch_size": args.batch_size},
    }
    if args.prediction_folder:
        evaluate_kwargs["prediction_folder"] = args.prediction_folder

    results = mteb.evaluate(model, **evaluate_kwargs)
    _write_summary(args.output_folder, args.model, tasks, results)
    return 0


def _load_mteb() -> Any:
    try:
        return importlib.import_module("mteb")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MTEB is not installed. Install benchmark dependencies with "
            '`python -m pip install -e ".[benchmark]"`.'
        ) from exc


def _load_model(args: argparse.Namespace, mteb: Any) -> Any:
    if args.model == "hash":
        return CamRAGMTEBModel(
            HashEmbeddingBackend(dim=args.hash_dim),
            name=f"cam-rag-hash-{args.hash_dim}",
        )
    return mteb.get_model(args.model)


def _load_tasks(args: argparse.Namespace, mteb: Any) -> Any:
    if args.tasks:
        return mteb.get_tasks(tasks=args.tasks, languages=args.languages)

    benchmark = mteb.get_benchmark(args.benchmark)
    if hasattr(mteb, "filter_tasks"):
        return mteb.filter_tasks(benchmark, task_types=[args.task_type])

    return [
        task
        for task in benchmark.tasks
        if getattr(getattr(task, "metadata", None), "type", None) == args.task_type
    ]


def _coerce_texts(inputs: Any) -> list[str]:
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, dict):
        return [_input_to_text(inputs)]

    texts: list[str] = []
    for item in _safe_iter(inputs):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(_input_to_text(item))
        elif _looks_like_batch(item):
            texts.extend(_coerce_texts(item))
        else:
            texts.append(str(item))
    return texts


def _safe_iter(inputs: Any) -> Iterable[Any]:
    try:
        return iter(inputs)
    except TypeError:
        return iter([inputs])


def _looks_like_batch(item: Any) -> bool:
    return not isinstance(item, (str, bytes, dict)) and hasattr(item, "__iter__")


def _input_to_text(item: dict[str, Any]) -> str:
    for key in ("text", "sentence", "query", "document", "passage"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return " ".join(str(value) for value in item.values() if value is not None)


def _write_summary(output_folder: str, model_name: str, tasks: Any, results: Any) -> None:
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    task_names = [
        getattr(getattr(task, "metadata", None), "name", str(task))
        for task in list(tasks)
    ]
    summary = {
        "model": model_name,
        "tasks": task_names,
        "result_count": len(results) if hasattr(results, "__len__") else None,
    }
    (output_path / "cam_rag_mteb_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
