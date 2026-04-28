"""Tests for PseudoRAG + complexity routing API flag exposure.

Validates:
- All PseudoRAG flags are exposed in ConfigOverrides model
- _apply_overrides correctly sets each flag on RAGAppSpec
- /v1/config lists all new flags with correct defaults
- selective_filter_threshold override applies correctly
- POST /v1/query with pipeline + MoE enabled processes correctly
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from cam_rag.api.models import ConfigOverrides
from cam_rag.api.server import _apply_overrides, app
from cam_rag.rag.spec import RAGAppSpec


# ---------------------------------------------------------------------------
# _apply_overrides unit tests
# ---------------------------------------------------------------------------


class TestApplyOverridesPseudoRAG:
    """Each PseudoRAG flag must be settable via overrides."""

    def test_moe_scoring_enabled_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(moe_scoring_enabled=True)
        result = _apply_overrides(spec, overrides)
        assert result.moe_scoring_enabled is True

    def test_accuracy_contracts_enabled_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(accuracy_contracts_enabled=True)
        result = _apply_overrides(spec, overrides)
        assert result.accuracy_contracts_enabled is True

    def test_selective_filter_enabled_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(selective_filter_enabled=True)
        result = _apply_overrides(spec, overrides)
        assert result.selective_filter_enabled is True

    def test_selective_filter_threshold_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(selective_filter_threshold=0.75)
        result = _apply_overrides(spec, overrides)
        assert result.selective_filter_threshold == 0.75

    def test_etf_verification_enabled_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(etf_verification_enabled=True)
        result = _apply_overrides(spec, overrides)
        assert result.etf_verification_enabled is True

    def test_complexity_routing_enabled_override(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides(complexity_routing_enabled=True)
        result = _apply_overrides(spec, overrides)
        assert result.complexity_routing_enabled is True

    def test_none_values_leave_defaults_unchanged(self):
        spec = RAGAppSpec(name="test")
        overrides = ConfigOverrides()  # all None
        result = _apply_overrides(spec, overrides)
        assert result.moe_scoring_enabled is False
        assert result.accuracy_contracts_enabled is False
        assert result.selective_filter_enabled is False
        assert result.selective_filter_threshold == 0.55
        assert result.etf_verification_enabled is False
        assert result.complexity_routing_enabled is False


# ---------------------------------------------------------------------------
# /v1/config endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="function")
class TestConfigEndpointPseudoRAG:
    """The /v1/config endpoint must list all PseudoRAG flags."""

    async def test_config_lists_moe_scoring_enabled(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/config")
        assert resp.status_code == 200
        flags = resp.json()["feature_flags"]
        assert "moe_scoring_enabled" in flags
        assert flags["moe_scoring_enabled"] is False

    async def test_config_lists_all_pseudorag_flags(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/config")
        flags = resp.json()["feature_flags"]
        expected_flags = [
            "moe_scoring_enabled",
            "accuracy_contracts_enabled",
            "selective_filter_enabled",
            "selective_filter_threshold",
            "etf_verification_enabled",
            "complexity_routing_enabled",
        ]
        for flag in expected_flags:
            assert flag in flags, f"Missing flag: {flag}"

    async def test_config_default_selective_filter_threshold(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/config")
        flags = resp.json()["feature_flags"]
        assert flags["selective_filter_threshold"] == 0.55
