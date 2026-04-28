"""Tests for MTEB benchmark integration with PseudoRAG spec overrides.

Validates:
- --spec-overrides flag parses correctly
- Summary JSON includes spec_overrides field
- Hash backend + MoE pipeline config
- Benchmark result labels differentiate MoE-enabled vs disabled runs
- Regression tolerance applies across PseudoRAG arms
- Invalid JSON is rejected
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cam_rag.benchmarks.mteb import (
    _write_summary,
    build_arg_parser,
    parse_spec_overrides,
)
from cam_rag.rl.feedback import DEFAULT_ARMS


# ---------------------------------------------------------------------------
# parse_spec_overrides tests
# ---------------------------------------------------------------------------


class TestParseSpecOverrides:
    """The --spec-overrides parser must handle valid and invalid input."""

    def test_none_returns_empty_dict(self):
        assert parse_spec_overrides(None) == {}

    def test_empty_string_returns_empty_dict(self):
        assert parse_spec_overrides("") == {}

    def test_valid_json_returns_dict(self):
        result = parse_spec_overrides('{"moe_scoring_enabled": true}')
        assert result == {"moe_scoring_enabled": True}

    def test_multiple_flags_parsed(self):
        raw = '{"moe_scoring_enabled": true, "selective_filter_threshold": 0.7}'
        result = parse_spec_overrides(raw)
        assert result["moe_scoring_enabled"] is True
        assert result["selective_filter_threshold"] == 0.7

    def test_invalid_json_raises_system_exit(self):
        with pytest.raises(SystemExit, match="not valid JSON"):
            parse_spec_overrides("{invalid")

    def test_non_dict_json_raises_system_exit(self):
        with pytest.raises(SystemExit, match="must be a JSON object"):
            parse_spec_overrides("[1, 2, 3]")


# ---------------------------------------------------------------------------
# CLI argument tests
# ---------------------------------------------------------------------------


class TestSpecOverridesArgParsing:
    """The arg parser must accept and propagate --spec-overrides."""

    def test_arg_parser_accepts_spec_overrides(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--spec-overrides", '{"moe_scoring_enabled": true}'])
        assert args.spec_overrides == '{"moe_scoring_enabled": true}'

    def test_arg_parser_default_spec_overrides_is_none(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.spec_overrides is None


# ---------------------------------------------------------------------------
# Summary JSON includes spec_overrides
# ---------------------------------------------------------------------------


class _FakeTask:
    """Minimal task object with metadata.name."""

    def __init__(self, name: str):
        self.metadata = type("M", (), {"name": name})()


class TestSummaryWithSpecOverrides:
    """_write_summary must include spec_overrides in output JSON."""

    def test_summary_includes_spec_overrides(self, tmp_path: Path):
        overrides = {"moe_scoring_enabled": True, "etf_verification_enabled": True}
        _write_summary(
            str(tmp_path),
            "cam-rag-hash-256",
            [_FakeTask("SciFactRetrieval")],
            [],
            spec_overrides=overrides,
        )
        summary_path = tmp_path / "cam_rag_mteb_summary.json"
        data = json.loads(summary_path.read_text())
        assert "spec_overrides" in data
        assert data["spec_overrides"]["moe_scoring_enabled"] is True
        assert data["spec_overrides"]["etf_verification_enabled"] is True

    def test_summary_omits_spec_overrides_when_empty(self, tmp_path: Path):
        _write_summary(
            str(tmp_path),
            "cam-rag-hash-256",
            [_FakeTask("SciFactRetrieval")],
            [],
            spec_overrides={},
        )
        summary_path = tmp_path / "cam_rag_mteb_summary.json"
        data = json.loads(summary_path.read_text())
        assert "spec_overrides" not in data

    def test_summary_differentiates_moe_enabled_vs_disabled(self, tmp_path: Path):
        """Two runs with different overrides produce distinguishable summaries."""
        moe_dir = tmp_path / "moe_enabled"
        moe_dir.mkdir()
        _write_summary(
            str(moe_dir),
            "cam-rag-hash-256",
            [_FakeTask("SciFactRetrieval")],
            [],
            spec_overrides={"moe_scoring_enabled": True},
        )

        base_dir = tmp_path / "baseline"
        base_dir.mkdir()
        _write_summary(
            str(base_dir),
            "cam-rag-hash-256",
            [_FakeTask("SciFactRetrieval")],
            [],
            spec_overrides=None,
        )

        moe_data = json.loads((moe_dir / "cam_rag_mteb_summary.json").read_text())
        base_data = json.loads((base_dir / "cam_rag_mteb_summary.json").read_text())

        assert "spec_overrides" in moe_data
        assert "spec_overrides" not in base_data


# ---------------------------------------------------------------------------
# PseudoRAG arms coverage
# ---------------------------------------------------------------------------


class TestPseudoRAGArmsCoverage:
    """Benchmark arms must cover PseudoRAG technique subsets."""

    def test_moe_scored_arm_techniques(self):
        techniques = DEFAULT_ARMS["moe_scored"]
        assert "moe_importance_scoring" in techniques
        assert "selective_filter" in techniques

    def test_full_pseudorag_arm_complete(self):
        techniques = DEFAULT_ARMS["full_pseudorag"]
        expected = {
            "moe_importance_scoring",
            "accuracy_contracts",
            "selective_filter",
            "etf_verification",
        }
        assert expected.issubset(techniques)
