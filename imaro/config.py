"""IMARO configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from imaro.providers.base import LLMProvider
from imaro.providers.claude_api import ClaudeAPIProvider

if TYPE_CHECKING:
    from imaro.execution.base import Executor


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    type: str = "claude"  # claude | gemini
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-20250514"


class IMAROConfig(BaseModel):
    """Top-level configuration for an IMARO run."""

    # Provider configurations keyed by role
    providers: dict[str, ProviderConfig] = Field(default_factory=lambda: {
        "refiner": ProviderConfig(),
        "planner": ProviderConfig(),
        "consensus": ProviderConfig(),
        "milestone_generator": ProviderConfig(),
        "reviewer": ProviderConfig(),
        "contradiction_detector": ProviderConfig(),
        "drift_detector": ProviderConfig(),
        "executor": ProviderConfig(),
    })

    # Executor type: "claude" (uses claude CLI) or "gemini" (uses Gemini API)
    executor_type: str = "gemini"

    # Thresholds
    consensus_threshold: float = 0.75
    intention_alignment_threshold: float = 0.70
    drift_alert_threshold: float = 0.4

    # Limits
    max_refinement_rounds: int = 3
    max_retries: int = 3
    plan_agents: int = 3
    max_fix_attempts: int = 3

    # Timeouts (seconds)
    execution_timeout_low: int = 300      # 5 min
    execution_timeout_medium: int = 600   # 10 min
    execution_timeout_high: int = 1200    # 20 min

    # Claude Code allowed tools
    claude_code_allowed_tools: list[str] = Field(default_factory=lambda: [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    ])

    def get_provider(self, role: str) -> LLMProvider:
        """Instantiate and return an LLMProvider for the given role."""
        cfg = self.providers.get(role)
        if cfg is None:
            cfg = ProviderConfig()

        if cfg.type == "claude":
            api_key = os.environ.get(cfg.api_key_env, "")
            return ClaudeAPIProvider(api_key=api_key, model=cfg.model)

        if cfg.type == "gemini":
            from imaro.providers.gemini_api import GeminiAPIProvider

            api_key = os.environ.get(cfg.api_key_env, "")
            return GeminiAPIProvider(api_key=api_key, model=cfg.model)

        raise ValueError(f"Unknown provider type: {cfg.type}")

    def get_executor(self) -> Executor:
        """Instantiate and return the configured Executor."""
        if self.executor_type == "claude":
            from imaro.execution.claude_code import ClaudeCodeExecutor

            return ClaudeCodeExecutor(
                allowed_tools=self.claude_code_allowed_tools,
            )

        if self.executor_type == "gemini":
            from imaro.execution.gemini_executor import GeminiExecutor

            provider = self.get_provider("executor")
            return GeminiExecutor(provider=provider)

        raise ValueError(f"Unknown executor type: {self.executor_type}")

    def get_execution_timeout(self, complexity: str) -> int:
        """Return timeout in seconds for a given complexity level."""
        return {
            "low": self.execution_timeout_low,
            "medium": self.execution_timeout_medium,
            "high": self.execution_timeout_high,
        }.get(complexity, self.execution_timeout_medium)
