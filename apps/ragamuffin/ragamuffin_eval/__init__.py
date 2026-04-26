"""Ragamuffin evaluation fixture helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cam_rag.evaluation import GoldenQuestion, load_golden_questions_jsonl


@dataclass(frozen=True, slots=True)
class RagamuffinEvalFixture:
    docs_dir: Path
    golden_path: Path
    cases: list[GoldenQuestion]


def load_eval_fixture(root: str | Path) -> RagamuffinEvalFixture:
    """Load a Ragamuffin document-folder sample evaluation fixture."""

    sample_root = Path(root)
    docs_dir = sample_root / "docs"
    golden_path = sample_root / "golden_questions.jsonl"
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"missing sample docs directory: {docs_dir}")
    if not golden_path.is_file():
        raise FileNotFoundError(f"missing golden questions file: {golden_path}")
    return RagamuffinEvalFixture(
        docs_dir=docs_dir,
        golden_path=golden_path,
        cases=load_golden_questions_jsonl(golden_path),
    )
