"""MTEB benchmark runner for CAM RAG embedding backends."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cam_rag.retrieval import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    OllamaEmbeddingBackend,
    OpenRouterEmbeddingBackend,
)


class CamRAGMTEBModel:
    """MTEB-compatible wrapper around a `cam_rag` embedding backend.

    Implements the MTEB ``EncoderProtocol`` (encode + similarity +
    similarity_pairwise + mteb_model_meta) so the model passes the
    ``isinstance(model, EncoderProtocol)`` runtime check in mteb >= 2.12.
    """

    def __init__(self, backend: EmbeddingBackend, *, name: str = "cam-rag-hash") -> None:
        self.backend = backend
        self.mteb_model_meta = {"name": name, "revision": None, "release_date": None}

    def encode(self, inputs: Any, **_: Any) -> Any:
        """Encode MTEB inputs using the wrapped backend.

        MTEB has supported multiple encode signatures over time. This wrapper
        accepts a plain list of strings as well as iterable batches from the
        current DataLoader-based protocol.

        Returns a numpy array when numpy is available (required by MTEB
        retrieval evaluator), otherwise a nested list.
        """
        texts = _coerce_texts(inputs)
        if hasattr(self.backend, "embed_batch"):
            vectors = self.backend.embed_batch(texts)
        else:
            vectors = [self.backend.embed(text) for text in texts]
        return _to_array(vectors)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
        """Cosine similarity matrix between two embedding sets."""
        np = _import_numpy()
        e1 = np.asarray(embeddings1, dtype=np.float32)
        e2 = np.asarray(embeddings2, dtype=np.float32)
        n1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12, None)
        n2 = e2 / np.clip(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12, None)
        return n1 @ n2.T

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> Any:
        """Pairwise cosine similarity between corresponding embeddings."""
        np = _import_numpy()
        e1 = np.asarray(embeddings1, dtype=np.float32)
        e2 = np.asarray(embeddings2, dtype=np.float32)
        n1 = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12, None)
        n2 = e2 / np.clip(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12, None)
        return np.sum(n1 * n2, axis=1)


def _import_numpy() -> Any:
    """Import numpy lazily (it's a transitive dep of mteb)."""
    return importlib.import_module("numpy")


def _to_array(vectors: list[list[float]]) -> Any:
    """Convert embedding list to numpy array when available."""
    try:
        np = _import_numpy()
        return np.asarray(vectors, dtype=np.float32)
    except (ImportError, ModuleNotFoundError):
        return vectors


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
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help=(
            "Explicit embedding dimension for backends that need it. "
            "Defaults: OpenRouter pplx-embed-v1-4b=2560, Ollama bge-m3=1024."
        ),
    )
    parser.add_argument(
        "--spec-overrides",
        default=None,
        help=(
            "JSON string of RAGAppSpec flag overrides to label this benchmark run "
            '(e.g. \'{"moe_scoring_enabled": true}\').'
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mteb = _load_mteb()
    model = _load_model(args, mteb)
    tasks = _load_tasks(args, mteb)

    spec_overrides = parse_spec_overrides(args.spec_overrides)

    evaluate_kwargs: dict[str, Any] = {
        "tasks": tasks,
        "encode_kwargs": {"batch_size": args.batch_size},
        "overwrite_strategy": "always",
    }
    if args.prediction_folder:
        evaluate_kwargs["prediction_folder"] = args.prediction_folder

    model_result = mteb.evaluate(model, **evaluate_kwargs)
    _write_summary(
        args.output_folder,
        args.model,
        tasks,
        model_result,
        spec_overrides=spec_overrides,
    )
    return 0


def parse_spec_overrides(raw: str | None) -> dict[str, Any]:
    """Parse a JSON string of RAGAppSpec overrides.

    Returns an empty dict when *raw* is None or empty.
    Raises ``SystemExit`` on invalid JSON.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise SystemExit("--spec-overrides must be a JSON object (dict).")
        return parsed
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--spec-overrides is not valid JSON: {exc}") from exc


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

    if args.model.startswith("openrouter:"):
        model_id = args.model[len("openrouter:"):]
        dim = args.embedding_dim or 2560
        backend = OpenRouterEmbeddingBackend(model=model_id, dim=dim)
        return CamRAGMTEBModel(backend, name=f"openrouter-{model_id}")

    if args.model.startswith("ollama:"):
        model_id = args.model[len("ollama:"):]
        dim = args.embedding_dim or 1024
        backend = OllamaEmbeddingBackend(model=model_id, dim=dim)
        return CamRAGMTEBModel(backend, name=f"ollama-{model_id}")

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
        # MTEB DataLoader yields BatchedInput dicts where 'text' is a list
        batch_texts = _extract_batch_texts(inputs)
        if batch_texts is not None:
            return batch_texts
        return [_input_to_text(inputs)]

    texts: list[str] = []
    for item in _safe_iter(inputs):
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            batch_texts = _extract_batch_texts(item)
            if batch_texts is not None:
                texts.extend(batch_texts)
            else:
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


def _extract_batch_texts(item: dict[str, Any]) -> list[str] | None:
    """Extract a list of strings from a batched MTEB dict.

    MTEB's DataLoader yields ``BatchedInput`` dicts where the text key
    contains a *list* of strings (one per sample in the batch).  When such
    a structure is detected, return the list; otherwise return ``None`` so
    the caller can fall back to single-item extraction.
    """
    for key in ("text", "sentence", "query", "document", "passage"):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
            return list(value)
    return None


def _input_to_text(item: dict[str, Any]) -> str:
    for key in ("text", "sentence", "query", "document", "passage"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return " ".join(str(value) for value in item.values() if value is not None)


def _write_summary(
    output_folder: str,
    model_name: str,
    tasks: Any,
    results: Any,
    *,
    spec_overrides: dict[str, Any] | None = None,
) -> None:
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    task_names = [
        getattr(getattr(task, "metadata", None), "name", str(task))
        for task in list(tasks)
    ]

    # Extract per-task scores from ModelResult.task_results
    scores: dict[str, Any] = {}
    task_results = getattr(results, "task_results", None)
    if task_results:
        for tr in task_results:
            name = getattr(tr, "task_name", None) or str(tr)
            main_score = getattr(tr, "main_score", None)
            if main_score is not None:
                scores[name] = main_score
            elif hasattr(tr, "scores"):
                # Fallback: extract from scores dict
                for split, split_scores in tr.scores.items():
                    if split_scores:
                        for entry in split_scores:
                            if "main_score" in entry:
                                scores[f"{name}_{split}"] = entry["main_score"]

    result_count = len(task_results) if task_results else (
        len(results) if hasattr(results, "__len__") else None
    )
    summary: dict[str, Any] = {
        "model": model_name,
        "tasks": task_names,
        "result_count": result_count,
    }
    if scores:
        summary["scores"] = scores
    if spec_overrides:
        summary["spec_overrides"] = spec_overrides
    (output_path / "cam_rag_mteb_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
