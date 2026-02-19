"""Tests for IMAROConfig."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imaro.config import IMAROConfig, ProviderConfig
from imaro.providers.claude_api import ClaudeAPIProvider


class TestIMAROConfig:
    def test_default_values(self):
        cfg = IMAROConfig()
        assert cfg.consensus_threshold == 0.75
        assert cfg.intention_alignment_threshold == 0.70
        assert cfg.drift_alert_threshold == 0.4
        assert cfg.max_refinement_rounds == 3
        assert cfg.max_retries == 3
        assert cfg.plan_agents == 3
        assert cfg.max_fix_attempts == 3
        assert cfg.execution_timeout_low == 300
        assert cfg.execution_timeout_medium == 600
        assert cfg.execution_timeout_high == 1200
        assert "Read" in cfg.claude_code_allowed_tools
        assert "Write" in cfg.claude_code_allowed_tools
        assert cfg.executor_type == "claude"
        assert cfg.max_milestones == 5

    def test_default_provider_roles(self):
        cfg = IMAROConfig()
        expected_roles = [
            "refiner", "planner", "consensus",
            "milestone_generator", "reviewer",
            "contradiction_detector", "drift_detector",
            "executor",
        ]
        for role in expected_roles:
            assert role in cfg.providers

    def test_get_provider_returns_gemini_api_provider(self):
        from imaro.providers.gemini_api import GeminiAPIProvider

        cfg = IMAROConfig()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = cfg.get_provider("refiner")
        assert isinstance(provider, GeminiAPIProvider)

    def test_get_provider_gemini_type(self):
        from imaro.providers.gemini_api import GeminiAPIProvider

        cfg = IMAROConfig(
            providers={
                "test": ProviderConfig(
                    type="gemini",
                    api_key_env="GOOGLE_API_KEY",
                    model="gemini-2.5-flash",
                )
            }
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = cfg.get_provider("test")
        assert isinstance(provider, GeminiAPIProvider)
        assert provider.get_model_name() == "gemini-2.5-flash"

    def test_get_provider_unknown_type_raises(self):
        cfg = IMAROConfig(
            providers={"custom": ProviderConfig(type="unknown_provider")}
        )
        with pytest.raises(ValueError, match="Unknown provider type"):
            cfg.get_provider("custom")

    def test_get_provider_missing_role_uses_default(self):
        from imaro.providers.gemini_api import GeminiAPIProvider

        cfg = IMAROConfig()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = cfg.get_provider("nonexistent_role")
        assert isinstance(provider, GeminiAPIProvider)

    def test_get_execution_timeout_low(self):
        cfg = IMAROConfig()
        assert cfg.get_execution_timeout("low") == 300

    def test_get_execution_timeout_medium(self):
        cfg = IMAROConfig()
        assert cfg.get_execution_timeout("medium") == 600

    def test_get_execution_timeout_high(self):
        cfg = IMAROConfig()
        assert cfg.get_execution_timeout("high") == 1200

    def test_get_execution_timeout_unknown_defaults_to_medium(self):
        cfg = IMAROConfig()
        assert cfg.get_execution_timeout("unknown") == 600

    def test_custom_timeout_values(self):
        cfg = IMAROConfig(
            execution_timeout_low=100,
            execution_timeout_medium=200,
            execution_timeout_high=400,
        )
        assert cfg.get_execution_timeout("low") == 100
        assert cfg.get_execution_timeout("medium") == 200
        assert cfg.get_execution_timeout("high") == 400


class TestGetExecutor:
    def test_get_executor_claude(self):
        from imaro.execution.claude_code import ClaudeCodeExecutor

        cfg = IMAROConfig()  # default is now "claude"
        executor = cfg.get_executor()
        assert isinstance(executor, ClaudeCodeExecutor)

    def test_get_executor_gemini(self):
        from imaro.execution.gemini_executor import GeminiExecutor

        cfg = IMAROConfig(
            executor_type="gemini",
            providers={
                "executor": ProviderConfig(
                    type="gemini",
                    api_key_env="GOOGLE_API_KEY",
                    model="gemini-2.5-flash",
                ),
            },
        )
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            executor = cfg.get_executor()
        assert isinstance(executor, GeminiExecutor)

    def test_get_executor_unknown_type_raises(self):
        cfg = IMAROConfig(executor_type="unknown")
        with pytest.raises(ValueError, match="Unknown executor type"):
            cfg.get_executor()
