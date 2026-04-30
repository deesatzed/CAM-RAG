"""Fine-tuning orchestrator for domain-specific embedding adaptation.

Wraps ``SentenceTransformerTrainer`` with a configuration dataclass
and produces a ``FineTuneResult`` on completion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FineTuneConfig:
    """Configuration for an embedding fine-tuning run.

    Parameters
    ----------
    base_model_name:
        HuggingFace model identifier (user-selected, never hardcoded).
    output_dir:
        Directory where the fine-tuned model will be saved.
    epochs:
        Number of training epochs.
    batch_size:
        Per-device training batch size.
    learning_rate:
        Peak learning rate.
    warmup_ratio:
        Fraction of steps for linear warmup.
    max_seq_length:
        Maximum input token length.
    seed:
        Random seed for reproducibility.
    use_mps:
        Use Apple MPS accelerator when available.
    """

    base_model_name: str
    output_dir: str = "output/finetuned"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 256
    seed: int = 42
    use_mps: bool = True


@dataclass(slots=True)
class FineTuneResult:
    """Result from a fine-tuning run."""

    output_dir: str
    final_loss: float | None
    epochs_completed: int
    num_training_samples: int
    base_model: str


class FineTuneOrchestrator:
    """Orchestrate contrastive fine-tuning of a sentence-transformers model.

    Parameters
    ----------
    config:
        Fine-tuning configuration.
    """

    def __init__(self, config: FineTuneConfig) -> None:
        self.config = config

    def train(self, dataset: Any) -> FineTuneResult:
        """Fine-tune on a HuggingFace Dataset with anchor/positive/negative columns.

        Parameters
        ----------
        dataset:
            A ``datasets.Dataset`` with ``anchor``, ``positive``, ``negative``
            string columns (as produced by ``TrainingDataGenerator.generate_dataset()``).

        Returns
        -------
        FineTuneResult
            Result with output directory, final loss, and metadata.
        """
        try:
            from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
            from sentence_transformers.sentence_transformer.training_args import (
                SentenceTransformerTrainingArguments,
            )
            from sentence_transformers.sentence_transformer.losses import (
                MultipleNegativesRankingLoss,
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers>=5.0 is required for fine-tuning. "
                'Install with: pip install -e ".[finetune]"'
            ) from exc

        cfg = self.config
        device = _pick_device(cfg.use_mps)

        logger.info(
            "loading base model %s on %s",
            cfg.base_model_name,
            device,
        )
        model = SentenceTransformer(cfg.base_model_name, device=device)
        if cfg.max_seq_length:
            model.max_seq_length = cfg.max_seq_length

        loss = MultipleNegativesRankingLoss(model)

        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            seed=cfg.seed,
            logging_steps=1,
            save_strategy="epoch",
            # Disable reporting to avoid dependency on wandb/tensorboard
            report_to="none",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            loss=loss,
        )

        logger.info(
            "starting training: %d samples, %d epochs, batch=%d, lr=%s",
            len(dataset),
            cfg.epochs,
            cfg.batch_size,
            cfg.learning_rate,
        )
        train_result = trainer.train()

        # Save final model
        model.save(cfg.output_dir)
        logger.info("model saved to %s", cfg.output_dir)

        # Extract final loss from training history
        final_loss = _extract_final_loss(train_result)

        return FineTuneResult(
            output_dir=cfg.output_dir,
            final_loss=final_loss,
            epochs_completed=cfg.epochs,
            num_training_samples=len(dataset),
            base_model=cfg.base_model_name,
        )


def _pick_device(use_mps: bool) -> str:
    """Select the best device for training."""
    if use_mps:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except (ImportError, AttributeError):
            pass
    return "cpu"


def _extract_final_loss(train_result: Any) -> float | None:
    """Extract the final training loss from the trainer result."""
    if hasattr(train_result, "training_loss"):
        return float(train_result.training_loss)
    if hasattr(train_result, "metrics"):
        loss = train_result.metrics.get("train_loss")
        if loss is not None:
            return float(loss)
    return None
