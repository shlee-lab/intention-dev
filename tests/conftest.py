"""Shared fixtures for IMARO tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from imaro.config import IMAROConfig
from imaro.execution.claude_code import ClaudeCodeExecutor
from imaro.models.schemas import (
    ConsensusResult,
    IntentionDocument,
    IntentionMapping,
    LLMResponse,
    Milestone,
    Plan,
    UserReviewDecision,
)
from imaro.orchestrator import Orchestrator
from imaro.providers.base import LLMProvider
from imaro.ui.terminal import TerminalUI


@pytest.fixture
def mock_provider():
    """AsyncMock of LLMProvider that returns a configurable LLMResponse."""
    provider = AsyncMock(spec=LLMProvider)
    provider.generate.return_value = LLMResponse(
        content="{}",
        model="mock-model",
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    provider.get_model_name.return_value = "mock-model"
    return provider


@pytest.fixture
def sample_intention():
    """Fully populated IntentionDocument."""
    return IntentionDocument(
        purpose="Build a task management CLI tool",
        target_users="Software developers",
        core_value="Simple, fast task tracking from the terminal",
        functional_requirements=[
            "Create tasks with title and description",
            "List tasks with filtering",
            "Mark tasks as complete",
        ],
        non_functional_requirements=[
            "Response time under 100ms",
            "Works offline",
        ],
        out_of_scope=[
            "GUI interface",
            "Cloud sync",
        ],
        constraints=[
            "Python 3.11+",
            "No external database",
        ],
        success_criteria=[
            "User can create and complete tasks in under 5 seconds",
            "All data persists between sessions",
        ],
        raw_input="I want a simple CLI task manager",
    )


@pytest.fixture
def sample_milestone(sample_intention):
    """Milestone with IntentionMapping."""
    return Milestone(
        id="M1",
        name="Core Task CRUD",
        description="Implement create, read, update, delete for tasks",
        scope=["task creation", "task listing", "task completion"],
        constraints=["Use JSON file storage"],
        acceptance_criteria=[
            "Can create a task with title",
            "Can list all tasks",
            "Can mark a task complete",
        ],
        depends_on=[],
        estimated_complexity="medium",
        intention_mapping=IntentionMapping(
            fulfills_requirements=[
                "Create tasks with title and description",
                "List tasks with filtering",
                "Mark tasks as complete",
            ],
            serves_purpose=True,
            serves_core_value=True,
            success_criteria_addressed=[
                "User can create and complete tasks in under 5 seconds",
                "All data persists between sessions",
            ],
        ),
    )


@pytest.fixture
def sample_plan(sample_milestone):
    """Plan with milestones."""
    return Plan(
        name="Simplicity-First Plan",
        description="Minimal viable task manager",
        milestones=[sample_milestone],
        intention_alignment={
            "how_purpose_served": "Direct CLI task management",
            "how_core_value_delivered": "Fast terminal-native UX",
        },
        architecture_notes="Single-file JSON storage, typer CLI",
        tech_stack=["python", "typer", "json"],
    )


@pytest.fixture
def sample_consensus():
    """ConsensusResult with 'proceed' recommendation."""
    return ConsensusResult(
        plan_consensus_score=0.85,
        intention_alignment_score=0.90,
        overall_score=0.88,
        aligned_points=["All plans use typer", "JSON storage agreed"],
        divergent_points=["Disagreement on testing approach"],
        intention_gaps=[],
        recommendation="proceed",
    )


@pytest.fixture
def tmp_project(tmp_path):
    """tmp_path with .imaro/ dir pre-created."""
    imaro_dir = tmp_path / ".imaro"
    imaro_dir.mkdir()
    return tmp_path


# ── Integration-test fixtures ────────────────────────────────────────────────


@pytest.fixture
def mock_ui():
    """MagicMock of TerminalUI with all interactive methods pre-configured."""
    ui = MagicMock(spec=TerminalUI)

    # Approval gates
    ui.confirm_intention.return_value = True
    ui.confirm_milestones.return_value = True
    ui.present_review_report.return_value = UserReviewDecision.APPROVE

    # Refinement helpers
    ui.ask_refinement_questions.return_value = ["answer1"]
    ui.ask_skip_refinement.return_value = False

    # Display / no-op methods
    ui.show_progress.return_value = None
    ui.show_success.return_value = None
    ui.show_error.return_value = None
    ui.show_warning.return_value = None
    ui.display_intention.return_value = None
    ui.display_consensus.return_value = None
    ui.display_milestones.return_value = None

    return ui


@pytest.fixture
def mock_config(mock_provider):
    """IMAROConfig with get_provider patched to return mock_provider."""
    config = MagicMock(spec=IMAROConfig)
    config.get_provider.return_value = mock_provider
    config.plan_agents = 3
    config.consensus_threshold = 0.75
    config.max_refinement_rounds = 3
    config.max_retries = 3
    config.max_fix_attempts = 3
    config.max_milestones = 10
    config.drift_alert_threshold = 0.4
    config.claude_code_allowed_tools = [
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    ]
    config.executor_type = "claude"
    config.get_executor.return_value = ClaudeCodeExecutor(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    )
    return config


@pytest.fixture
def orchestrator(mock_config, mock_ui):
    """Orchestrator wired with mock config and mock UI."""
    return Orchestrator(config=mock_config, ui=mock_ui)
