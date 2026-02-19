"""Shared fixtures for IMARO tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from imaro.models.schemas import (
    ConsensusResult,
    IntentionDocument,
    IntentionMapping,
    LLMResponse,
    Milestone,
    Plan,
)
from imaro.providers.base import LLMProvider


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
