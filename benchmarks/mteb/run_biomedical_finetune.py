#!/usr/bin/env python3
"""Biomedical ensemble fine-tuning pipeline for MTEB benchmarks.

End-to-end workflow:
1. Load biomedical corpus from MTEB (NFCorpus) or local documents
2. Profile domain gap against a user-selected base model
3. Generate contrastive training triplets
4. Fine-tune the embedding model
5. Benchmark before/after with full-pipeline MTEB evaluation

Usage:
    python benchmarks/mteb/run_biomedical_finetune.py \\
        --base-model all-MiniLM-L6-v2 \\
        --output-dir ./output/biomedical-finetuned \\
        --corpus-source nfcorpus \\
        --epochs 3

    python benchmarks/mteb/run_biomedical_finetune.py \\
        --base-model all-MiniLM-L6-v2 \\
        --output-dir ./output/biomedical-finetuned \\
        --corpus-source local \\
        --corpus-dir ./data/pubmed_abstracts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model on biomedical data and benchmark via MTEB.",
    )
    parser.add_argument(
        "--base-model",
        default="all-MiniLM-L6-v2",
        help="Base sentence-transformers model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/biomedical-finetuned",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--corpus-source",
        choices=["nfcorpus", "local"],
        default="nfcorpus",
        help="Source of biomedical training data.",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Directory of local .txt/.md files when --corpus-source=local.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum training samples to generate.",
    )
    parser.add_argument(
        "--benchmark-tasks",
        nargs="*",
        default=["SciFact", "NFCorpus"],
        help="MTEB tasks for before/after comparison.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip corpus profiling step.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip MTEB benchmarking (just fine-tune).",
    )
    return parser


def _load_nfcorpus_chunks():
    """Load NFCorpus from MTEB as Chunk objects."""
    import importlib

    mteb = importlib.import_module("mteb")
    tasks = mteb.get_tasks(tasks=["NFCorpus"])
    task = tasks[0]
    task.load_data()

    from cam_rag.rag.models import Chunk

    chunks = []
    # Access corpus from the test split
    corpus = task.corpus.get("test", {})
    for doc_id, doc in corpus.items():
        text = doc.get("text", "")
        title = doc.get("title", "")
        full_text = f"{title}\n{text}".strip() if title else text
        if full_text:
            chunks.append(
                Chunk(
                    id=str(doc_id),
                    document_id=str(doc_id),
                    text=full_text,
                    title=title,
                )
            )
    return chunks


def _load_local_chunks(corpus_dir: str):
    """Load local text files as Chunk objects."""
    from cam_rag.rag.models import Chunk

    chunks = []
    corpus_path = Path(corpus_dir)
    for i, filepath in enumerate(sorted(corpus_path.glob("**/*.txt"))):
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(
                Chunk(
                    id=f"local_{i}",
                    document_id=f"local_{i}",
                    text=text,
                    title=filepath.stem,
                    source=str(filepath),
                )
            )
    for i, filepath in enumerate(sorted(corpus_path.glob("**/*.md")), start=len(chunks)):
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(
                Chunk(
                    id=f"local_{i}",
                    document_id=f"local_{i}",
                    text=text,
                    title=filepath.stem,
                    source=str(filepath),
                )
            )
    return chunks


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Step 1: Load corpus
    print(f"Loading corpus from {args.corpus_source}...")
    if args.corpus_source == "nfcorpus":
        chunks = _load_nfcorpus_chunks()
    else:
        if not args.corpus_dir:
            print("Error: --corpus-dir required when --corpus-source=local")
            return 1
        chunks = _load_local_chunks(args.corpus_dir)

    print(f"  Loaded {len(chunks)} chunks")
    if not chunks:
        print("Error: No chunks loaded")
        return 1

    # Step 2: Profile domain gap
    if not args.skip_profile:
        from cam_rag.finetuning.backend import LocalSentenceTransformerBackend
        from cam_rag.finetuning.profiler import CorpusProfiler, DomainGapScorer

        print(f"\nProfiling domain gap against {args.base_model}...")
        backend = LocalSentenceTransformerBackend(args.base_model)
        profiler = CorpusProfiler(backend)
        profile = profiler.profile(chunks)

        scorer = DomainGapScorer()
        gap = scorer.score(profile)
        print(f"  Domain gap score: {gap.gap_score:.3f}")
        print(f"  Recommendation: {gap.recommendation}")
        for signal, value in gap.signal_scores.items():
            print(f"    {signal}: {value:.3f}")
    else:
        print("\nSkipping corpus profiling")

    # Step 3: Generate training data
    from cam_rag.finetuning.data import TrainingDataGenerator

    print(f"\nGenerating training data (max {args.max_samples} samples)...")
    generator = TrainingDataGenerator()
    dataset = generator.generate(chunks, max_samples=args.max_samples)
    print(f"  Generated {len(dataset)} training samples")

    # Step 4: Fine-tune
    from cam_rag.finetuning.trainer import FineTuneOrchestrator

    print(f"\nFine-tuning {args.base_model}...")
    output_path = Path(args.output_dir)
    orchestrator = FineTuneOrchestrator(
        base_model=args.base_model,
        output_dir=str(output_path),
    )
    result = orchestrator.train(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"  Training complete")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Model saved to: {output_path}")

    # Step 5: Benchmark before/after
    if not args.skip_benchmark:
        print(f"\nRunning MTEB benchmark comparison...")
        from benchmarks.mteb.run_finetune_comparison import main as comparison_main

        comparison_argv = [
            "--base-model", f"local:{args.base_model}",
            "--finetuned-model", f"local:{output_path}",
            "--tasks", *args.benchmark_tasks,
        ]
        comparison_main(comparison_argv)
    else:
        print("\nSkipping MTEB benchmark")

    # Save metadata
    metadata = {
        "base_model": args.base_model,
        "corpus_source": args.corpus_source,
        "num_chunks": len(chunks),
        "num_training_samples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    meta_path = output_path / "finetune_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nMetadata saved to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
