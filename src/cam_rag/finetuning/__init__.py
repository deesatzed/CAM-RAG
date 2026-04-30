"""Domain-specific embedding fine-tuning for CAM RAG.

Provides corpus profiling, contrastive training data generation,
fine-tuning orchestration, and a local sentence-transformers backend.
"""

from cam_rag.finetuning.backend import LocalSentenceTransformerBackend
from cam_rag.finetuning.data import TrainingDataGenerator
from cam_rag.finetuning.profiler import (
    CorpusProfile,
    CorpusProfiler,
    DomainGapReport,
    DomainGapScorer,
)
from cam_rag.finetuning.trainer import FineTuneConfig, FineTuneOrchestrator, FineTuneResult

__all__ = [
    "CorpusProfile",
    "CorpusProfiler",
    "DomainGapReport",
    "DomainGapScorer",
    "FineTuneConfig",
    "FineTuneOrchestrator",
    "FineTuneResult",
    "LocalSentenceTransformerBackend",
    "TrainingDataGenerator",
]
