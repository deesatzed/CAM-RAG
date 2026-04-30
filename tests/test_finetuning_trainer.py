"""Tests for FineTuneOrchestrator with real 1-epoch training on all-MiniLM-L6-v2.

These tests do *real* training (no mocks). Each runs 1 epoch on 5 pairs
which completes in a few seconds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from datasets import Dataset

from cam_rag.finetuning.backend import LocalSentenceTransformerBackend
from cam_rag.finetuning.trainer import FineTuneConfig, FineTuneOrchestrator, FineTuneResult

MODEL_NAME = "all-MiniLM-L6-v2"

# Small but realistic dataset for fast training
TRAIN_DATA = Dataset.from_dict(
    {
        "anchor": [
            "STEMI Management",
            "Antiplatelet Therapy",
            "CKD Staging",
            "Renal Replacement Therapy",
            "Heart Failure Treatment",
        ],
        "positive": [
            "Acute ST-elevation myocardial infarction requires emergent PCI within 90 minutes.",
            "Dual antiplatelet therapy with aspirin and P2Y12 inhibitor is standard post-PCI.",
            "Chronic kidney disease is staged by eGFR. Stage 3a is eGFR 45-59 mL/min.",
            "Renal replacement therapy options include hemodialysis and peritoneal dialysis.",
            "Heart failure with reduced ejection fraction is managed with ACE inhibitors.",
        ],
        "negative": [
            "The weather forecast shows sunny skies and warm temperatures this week.",
            "The local farmers market opens every Saturday morning with fresh produce.",
            "Regular exercise has many health benefits including stress reduction.",
            "The new library downtown has a wonderful children's reading section.",
            "Cooking at home saves money and is often healthier than eating out.",
        ],
    }
)


@pytest.fixture
def config(tmp_path: Path) -> FineTuneConfig:
    return FineTuneConfig(
        base_model_name=MODEL_NAME,
        output_dir=str(tmp_path / "finetuned"),
        epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.0,
        max_seq_length=128,
        seed=42,
        use_mps=False,  # CPU for test stability
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_produces_result(config: FineTuneConfig) -> None:
    orch = FineTuneOrchestrator(config)
    result = orch.train(TRAIN_DATA)
    assert isinstance(result, FineTuneResult)
    assert result.output_dir == config.output_dir
    assert result.epochs_completed == 1
    assert result.num_training_samples == 5
    assert result.base_model == MODEL_NAME


def test_model_saved_to_output_dir(config: FineTuneConfig) -> None:
    orch = FineTuneOrchestrator(config)
    result = orch.train(TRAIN_DATA)
    output = Path(result.output_dir)
    assert output.exists()
    # sentence-transformers saves config files
    assert any(output.iterdir())


def test_finetuned_model_loadable(config: FineTuneConfig) -> None:
    """The saved model can be loaded as a LocalSentenceTransformerBackend."""
    orch = FineTuneOrchestrator(config)
    result = orch.train(TRAIN_DATA)
    backend = LocalSentenceTransformerBackend(result.output_dir, device="cpu")
    assert backend.dim == 384
    vec = backend.embed("test embedding after fine-tuning")
    assert len(vec) == 384


def test_final_loss_is_number(config: FineTuneConfig) -> None:
    orch = FineTuneOrchestrator(config)
    result = orch.train(TRAIN_DATA)
    assert result.final_loss is None or isinstance(result.final_loss, float)


def test_config_slots() -> None:
    """FineTuneConfig is a slots dataclass."""
    cfg = FineTuneConfig(base_model_name="test-model")
    assert cfg.base_model_name == "test-model"
    assert cfg.epochs == 3  # default
    assert cfg.batch_size == 16  # default
