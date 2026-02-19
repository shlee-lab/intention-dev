"""Integration tests for the Orchestrator pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imaro.models.schemas import (
    ExecutionResult,
    IntentionDocument,
    IntentionMapping,
    LLMResponse,
    Milestone,
    MilestoneStatus,
    Plan,
    ReviewReport,
    ReviewResult,
    UserReviewDecision,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _intention_json() -> dict:
    """Return a minimal valid intention document dict."""
    return {
        "purpose": "Build a CLI tool",
        "target_users": "Developers",
        "core_value": "Speed",
        "functional_requirements": ["Create items", "List items"],
        "non_functional_requirements": ["Fast"],
        "out_of_scope": ["GUI"],
        "constraints": ["Python 3.12"],
        "success_criteria": ["Works in <1s"],
    }


def _plan_json(name: str = "Test Plan") -> dict:
    """Return a minimal valid plan dict."""
    return {
        "name": name,
        "description": "A test plan",
        "architecture_notes": "Simple",
        "tech_stack": ["python"],
        "intention_alignment": {"how_purpose_served": "directly"},
        "milestones": [
            {
                "id": "M1",
                "name": "Step 1",
                "description": "First step",
                "scope": ["core"],
                "constraints": [],
                "acceptance_criteria": ["Works"],
                "depends_on": [],
                "estimated_complexity": "low",
                "intention_mapping": {
                    "fulfills_requirements": ["Create items", "List items"],
                    "serves_purpose": True,
                    "serves_core_value": True,
                    "success_criteria_addressed": ["Works in <1s"],
                },
            }
        ],
    }


def _consensus_json(score: float = 0.9) -> dict:
    """Return a consensus evaluation dict."""
    return {
        "plan_consensus_score": score,
        "intention_alignment_score": score,
        "aligned_points": ["All agree"],
        "divergent_points": [],
        "intention_gaps": [],
    }


def _milestone_json() -> dict:
    """Return milestones generation response."""
    return {
        "milestones": [
            {
                "id": "M1",
                "name": "Step 1",
                "description": "First step",
                "scope": ["core"],
                "constraints": [],
                "acceptance_criteria": ["Works"],
                "depends_on": [],
                "estimated_complexity": "low",
                "intention_mapping": {
                    "fulfills_requirements": ["Create items", "List items"],
                    "serves_purpose": True,
                    "serves_core_value": True,
                    "success_criteria_addressed": ["Works in <1s"],
                },
            }
        ]
    }


def _review_json(passed: bool = True) -> dict:
    """Return a review result dict."""
    return {
        "summary": "Looks good",
        "passed": passed,
        "intention_alignment_score": 1.0,
        "intention_alignment_details": "Aligned",
        "issues": [],
    }


def _drift_json(score: float = 0.0) -> dict:
    """Return a drift report dict."""
    return {
        "drift_score": score,
        "drifted_areas": [] if score < 0.4 else ["area1"],
        "recommendations": [],
        "still_aligned": score < 0.4,
    }


def _llm(content: dict | str) -> LLMResponse:
    """Build an LLMResponse from a dict or string."""
    if isinstance(content, dict):
        content = json.dumps(content)
    return LLMResponse(content=content, model="mock", usage={})


def _mock_subprocess(success: bool = True) -> AsyncMock:
    """Return an AsyncMock mimicking asyncio.create_subprocess_exec."""
    proc = AsyncMock()
    if success:
        proc.communicate.return_value = (
            json.dumps({"session_id": "sess-1", "result": "done"}).encode(),
            b"",
        )
        proc.returncode = 0
    else:
        proc.communicate.return_value = (b"", b"execution error")
        proc.returncode = 1
    proc.kill = MagicMock()
    return proc


# ── Happy Path ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_pipeline_happy_path(orchestrator, mock_provider, mock_ui, tmp_path):
    """Full pipeline: refine → plan → consensus → milestones → execute → review → success."""
    # Sequence of LLM responses:
    # 1. Refiner: returns ready doc
    # 2-4. Planner: 3 plans
    # 5. Consensus: high scores
    # 6. Milestones: single milestone
    # 7. Reviewer (4 roles): clean reviews
    # 8. Contradiction detector: empty
    # 9. Drift: low score (skipped for 1 milestone)
    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
        # 4 reviewer calls + 1 contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    assert result.intention is not None
    assert result.intention.purpose == "Build a CLI tool"
    assert len(result.milestones) > 0
    assert result.error == ""

    # State files written
    state_dir = tmp_path / ".imaro"
    assert (state_dir / "intention.json").exists()
    assert (state_dir / "plans.json").exists()
    assert (state_dir / "consensus.json").exists()
    assert (state_dir / "milestones.json").exists()
    assert (state_dir / "result.json").exists()


# ── Intention Refinement Paths ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refinement_with_questions(orchestrator, mock_provider, mock_ui, tmp_path):
    """Refiner asks questions first, then produces a ready doc on second call."""
    questions_response = {
        "status": "questions",
        "questions": [
            {"question": "What language?", "context": "Need to know"},
        ],
    }

    mock_provider.generate.side_effect = [
        # Round 1: questions
        _llm(questions_response),
        # Round 2 (refine_with_answers): ready doc
        _llm({"status": "ready", "document": _intention_json()}),
        # Plans
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        # Consensus
        _llm(_consensus_json(0.9)),
        # Milestones
        _llm(_milestone_json()),
        # Reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    mock_ui.ask_refinement_questions.assert_called_once()


@pytest.mark.asyncio
async def test_refinement_user_rejects_then_approves(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """User rejects intention on first pass, approves after edit."""
    mock_ui.confirm_intention.side_effect = [False, True]
    mock_ui.edit_intention.return_value = IntentionDocument(**_intention_json(), raw_input="test")

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        # Plans
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        # Consensus
        _llm(_consensus_json(0.9)),
        # Milestones
        _llm(_milestone_json()),
        # Reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    assert mock_ui.confirm_intention.call_count == 2
    mock_ui.edit_intention.assert_called_once()


# ── Consensus Failure Paths ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consensus_below_threshold_proceed_anyway(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """Consensus below threshold, user chooses 'proceed_anyway'."""
    mock_ui.handle_consensus_failure.return_value = "proceed_anyway"

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        # Low consensus → recommendation != "proceed"
        _llm(_consensus_json(0.3)),
        # Milestones
        _llm(_milestone_json()),
        # Reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    mock_ui.handle_consensus_failure.assert_called_once()


@pytest.mark.asyncio
async def test_consensus_below_threshold_abort(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """Consensus below threshold, user chooses 'abort' → pipeline fails."""
    mock_ui.handle_consensus_failure.return_value = "abort"

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.3)),
    ]

    result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is False
    assert "abort" in result.error.lower() or "User chose to abort" in result.error


# ── Milestone Approval ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_milestones_not_approved_aborts(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """User rejects milestones → pipeline returns early with error."""
    mock_ui.confirm_milestones.return_value = False

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
    ]

    result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is False
    assert "not approved" in result.error.lower()


# ── Execution Failure ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_failure_stops_pipeline(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """Executor returns success=False → milestone FAILED, pipeline stops."""
    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
    ]

    with patch(
        "asyncio.create_subprocess_exec", return_value=_mock_subprocess(success=False)
    ):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is False
    assert "failed" in result.error.lower()
    assert len(result.milestone_results) == 1
    assert result.milestone_results[0].status == MilestoneStatus.FAILED


# ── Review Loop ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_fix_then_approve(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """First review → FIX → second review → APPROVE."""
    mock_ui.present_review_report.side_effect = [
        UserReviewDecision.FIX,
        UserReviewDecision.APPROVE,
    ]

    critical_review = {
        "summary": "Issues found",
        "passed": False,
        "intention_alignment_score": 0.8,
        "intention_alignment_details": "Mostly aligned",
        "issues": [
            {
                "description": "Missing error handling",
                "severity": "critical",
                "file": "main.py",
                "line": 10,
                "suggestion": "Add try/except",
                "intention_relevant": False,
            }
        ],
    }

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
        # First review round (4 reviewers + contradiction)
        _llm(critical_review),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
        # Second review round after fix (4 reviewers + contradiction)
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    assert mock_ui.present_review_report.call_count == 2


@pytest.mark.asyncio
async def test_review_abort(orchestrator, mock_provider, mock_ui, tmp_path):
    """User chooses ABORT during review → milestone FAILED."""
    mock_ui.present_review_report.return_value = UserReviewDecision.ABORT

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
        # Reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is False
    assert len(result.milestone_results) == 1
    assert result.milestone_results[0].status == MilestoneStatus.FAILED


# ── Drift Detection ─────────────────────────────────────────────────────────


def _two_milestone_json() -> dict:
    """Return a response with 2 milestones to trigger drift check."""
    base = _milestone_json()["milestones"][0]
    m2 = dict(base, id="M2", name="Step 2", description="Second step")
    return {"milestones": [base, m2]}


@pytest.mark.asyncio
async def test_drift_alert_user_continues(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """Drift detected after 2nd milestone, user continues."""
    mock_ui.handle_drift.return_value = "continue"

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_two_milestone_json()),
        # Milestone 1: reviews + contradiction (no drift check — first milestone)
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
        # Milestone 2: reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
        # Drift check after milestone 2: high score
        _llm(_drift_json(0.6)),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is True
    mock_ui.handle_drift.assert_called_once()


@pytest.mark.asyncio
async def test_drift_alert_user_aborts(
    orchestrator, mock_provider, mock_ui, tmp_path
):
    """Drift detected → user aborts → pipeline fails."""
    mock_ui.handle_drift.return_value = "abort"

    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_two_milestone_json()),
        # Milestone 1: reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
        # Milestone 2: reviews + contradiction
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
        # Drift check: high score
        _llm(_drift_json(0.6)),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        result = await orchestrator.run("Build me a CLI tool", tmp_path)

    assert result.success is False
    assert "drift" in result.error.lower()


# ── State Persistence ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_files_written(orchestrator, mock_provider, mock_ui, tmp_path):
    """After a full run, all expected state files exist in .imaro/."""
    mock_provider.generate.side_effect = [
        _llm({"status": "ready", "document": _intention_json()}),
        _llm(_plan_json("Plan A")),
        _llm(_plan_json("Plan B")),
        _llm(_plan_json("Plan C")),
        _llm(_consensus_json(0.9)),
        _llm(_milestone_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm(_review_json()),
        _llm({"contradictions": []}),
    ]

    with patch("asyncio.create_subprocess_exec", return_value=_mock_subprocess()):
        await orchestrator.run("Build me a CLI tool", tmp_path)

    state_dir = tmp_path / ".imaro"
    for name in ["intention", "plans", "consensus", "milestones", "result"]:
        path = state_dir / f"{name}.json"
        assert path.exists(), f"Missing state file: {name}.json"
        data = json.loads(path.read_text())
        assert data, f"State file {name}.json is empty"
