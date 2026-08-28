"""
DocumentProfile — user/AI-provided intelligence about a document.

Drives all downstream pipeline decisions: preprocessing depth, adapter strength,
retrieval thresholds, derivative computation, generation style.

Unified from v3 DocumentProfile + xAI MuskProfile.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentProfile:
    accuracy_importance: str = "medium"           # critical | high | medium | low
    complexity_level: str = "medium"              # low | medium | high
    update_frequency: str = "low"                 # static | low | medium | high
    domain: str = "general"
    likely_question_types: list[str] = field(default_factory=lambda: ["synthesis"])
    tolerance_for_hallucination: str = "medium"   # zero | low | medium | high
    priority: str = "balanced"                    # precision | balanced | speed
    mission_critical: bool = False

    def __post_init__(self):
        valid = {
            "accuracy_importance": ["critical", "high", "medium", "low"],
            "complexity_level": ["low", "medium", "high"],
            "update_frequency": ["static", "low", "medium", "high"],
            "tolerance_for_hallucination": ["zero", "low", "medium", "high"],
            "priority": ["precision", "balanced", "speed"],
        }
        for k, vals in valid.items():
            if getattr(self, k) not in vals:
                raise ValueError(f"Invalid {k}: {getattr(self, k)}")

    def to_config(self) -> dict:
        """Convert profile to concrete processing parameters."""
        return {
            "chunk_sizes": {"low": 180, "medium": 120, "high": 80}[self.complexity_level],
            "deriv_depth": {"low": 1, "medium": 2, "high": 3}[self.complexity_level],
            "adapter_strength": {"low": 0.15, "medium": 0.25, "high": 0.40}[self.complexity_level],
            "metadata_richness": {"low": 3, "medium": 6, "high": 10}[self.complexity_level],
            "retrieval_threshold": {
                "critical": 0.78, "high": 0.72, "medium": 0.65, "low": 0.55,
            }[self.accuracy_importance],
            "generation_strictness": {
                "zero": "strict", "low": "strict", "medium": "balanced", "high": "creative",
            }[self.tolerance_for_hallucination],
            "priority_bias": {
                "precision": {"deriv": 1.6, "level_balance": 0.8},
                "balanced":  {"deriv": 1.0, "level_balance": 1.0},
                "speed":     {"deriv": 0.6, "level_balance": 1.3},
            }[self.priority],
        }
