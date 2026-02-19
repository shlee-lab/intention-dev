"""Tests for PlanGenerator, ConsensusEvaluator, and MilestoneGenerator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from imaro.models.schemas import (
    ConsensusResult,
    IntentionDocument,
    IntentionMapping,
    LLMResponse,
    Milestone,
    Plan,
)
from imaro.planning.consensus import ConsensusEvaluator
from imaro.planning.milestone import MilestoneGenerator
from imaro.planning.plan_generator import PlanGenerator, extract_json


# ── extract_json (plan_generator module) ────────────────────────────────────


class TestExtractJsonPlanGenerator:
    def test_plain_json(self):
        data = {"name": "plan"}
        assert extract_json(json.dumps(data)) == data

    def test_code_fenced_json(self):
        data = {"name": "plan"}
        text = f"```json\n{json.dumps(data)}\n```"
        assert extract_json(text) == data

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_json("not json")


# ── PlanGenerator ───────────────────────────────────────────────────────────


def _make_plan_response(name: str = "Test Plan") -> LLMResponse:
    plan_data = {
        "name": name,
        "description": "A test plan",
        "architecture_notes": "Simple",
        "tech_stack": ["python"],
        "intention_alignment": {},
        "milestones": [
            {
                "id": "M1",
                "name": "Step 1",
                "description": "Do something",
                "scope": ["thing"],
                "constraints": [],
                "acceptance_criteria": ["it works"],
                "depends_on": [],
                "estimated_complexity": "low",
                "intention_mapping": {
                    "fulfills_requirements": [],
                    "serves_purpose": True,
                    "serves_core_value": True,
                    "success_criteria_addressed": [],
                },
            }
        ],
    }
    return LLMResponse(content=json.dumps(plan_data), model="mock")


class TestPlanGenerator:
    @pytest.mark.asyncio
    async def test_generate_plans_with_three_providers(self, sample_intention):
        providers = []
        for i in range(3):
            p = AsyncMock()
            p.generate.return_value = _make_plan_response(f"Plan {i}")
            providers.append(p)

        gen = PlanGenerator()
        plans = await gen.generate_plans(
            sample_intention, num_agents=3, providers=providers
        )

        assert len(plans) == 3
        assert all(isinstance(p, Plan) for p in plans)

    @pytest.mark.asyncio
    async def test_partial_failure_returns_remaining(self, sample_intention):
        good1 = AsyncMock()
        good1.generate.return_value = _make_plan_response("Good 1")

        bad = AsyncMock()
        bad.generate.side_effect = RuntimeError("LLM error")

        good2 = AsyncMock()
        good2.generate.return_value = _make_plan_response("Good 2")

        gen = PlanGenerator()
        plans = await gen.generate_plans(
            sample_intention, num_agents=3, providers=[good1, bad, good2]
        )

        assert len(plans) == 2

    @pytest.mark.asyncio
    async def test_all_agents_fail_raises(self, sample_intention):
        providers = []
        for _ in range(3):
            p = AsyncMock()
            p.generate.side_effect = RuntimeError("fail")
            providers.append(p)

        gen = PlanGenerator()
        with pytest.raises(RuntimeError, match="All plan generation agents failed"):
            await gen.generate_plans(
                sample_intention, num_agents=3, providers=providers
            )

    @pytest.mark.asyncio
    async def test_no_providers_no_default_raises(self, sample_intention):
        gen = PlanGenerator()
        with pytest.raises(ValueError, match="Either providers or default_provider"):
            await gen.generate_plans(sample_intention, num_agents=3)

    @pytest.mark.asyncio
    async def test_default_provider_used_when_no_providers(self, sample_intention):
        default = AsyncMock()
        default.generate.return_value = _make_plan_response("Default Plan")

        gen = PlanGenerator()
        plans = await gen.generate_plans(
            sample_intention, num_agents=3, default_provider=default
        )

        assert len(plans) == 3
        # default provider should have been called 3 times
        assert default.generate.call_count == 3


# ── ConsensusEvaluator ──────────────────────────────────────────────────────


class TestConsensusEvaluator:
    @pytest.mark.asyncio
    async def test_scores_above_threshold_proceed(
        self, sample_intention, sample_plan, mock_provider
    ):
        consensus_data = {
            "plan_consensus_score": 0.90,
            "intention_alignment_score": 0.85,
            "aligned_points": ["Agreement on arch"],
            "divergent_points": [],
            "intention_gaps": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(consensus_data), model="mock"
        )

        evaluator = ConsensusEvaluator()
        result = await evaluator.evaluate(
            [sample_plan], sample_intention, mock_provider, threshold=0.75
        )

        assert result.recommendation == "proceed"
        # overall = 0.4 * 0.90 + 0.6 * 0.85 = 0.36 + 0.51 = 0.87
        assert abs(result.overall_score - 0.87) < 0.001

    @pytest.mark.asyncio
    async def test_scores_below_threshold_minus_015_strongly_advise_against(
        self, sample_intention, sample_plan, mock_provider
    ):
        consensus_data = {
            "plan_consensus_score": 0.30,
            "intention_alignment_score": 0.30,
            "aligned_points": [],
            "divergent_points": ["Everything"],
            "intention_gaps": ["Major gaps"],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(consensus_data), model="mock"
        )

        evaluator = ConsensusEvaluator()
        result = await evaluator.evaluate(
            [sample_plan], sample_intention, mock_provider, threshold=0.75
        )

        # overall = 0.4 * 0.30 + 0.6 * 0.30 = 0.30 < 0.60
        assert result.recommendation == "strongly_advise_against"

    @pytest.mark.asyncio
    async def test_scores_between_thresholds_advise_against(
        self, sample_intention, sample_plan, mock_provider
    ):
        # threshold=0.75, threshold-0.15=0.60
        # Need overall >= 0.60 and < 0.75
        # overall = 0.4 * pc + 0.6 * ia
        # pc=0.70, ia=0.65 -> overall = 0.28 + 0.39 = 0.67
        consensus_data = {
            "plan_consensus_score": 0.70,
            "intention_alignment_score": 0.65,
            "aligned_points": [],
            "divergent_points": [],
            "intention_gaps": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(consensus_data), model="mock"
        )

        evaluator = ConsensusEvaluator()
        result = await evaluator.evaluate(
            [sample_plan], sample_intention, mock_provider, threshold=0.75
        )

        assert result.recommendation == "advise_against"

    @pytest.mark.asyncio
    async def test_overall_score_formula(
        self, sample_intention, sample_plan, mock_provider
    ):
        consensus_data = {
            "plan_consensus_score": 0.50,
            "intention_alignment_score": 1.0,
            "aligned_points": [],
            "divergent_points": [],
            "intention_gaps": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(consensus_data), model="mock"
        )

        evaluator = ConsensusEvaluator()
        result = await evaluator.evaluate(
            [sample_plan], sample_intention, mock_provider
        )

        # overall = 0.4 * 0.50 + 0.6 * 1.0 = 0.20 + 0.60 = 0.80
        assert abs(result.overall_score - 0.80) < 0.001


# ── MilestoneGenerator ──────────────────────────────────────────────────────


class TestMilestoneGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_milestones(
        self, sample_intention, sample_plan, sample_consensus, mock_provider
    ):
        milestones_data = {
            "milestones": [
                {
                    "id": "M1",
                    "name": "Core",
                    "description": "Core implementation",
                    "scope": ["task CRUD"],
                    "constraints": [],
                    "acceptance_criteria": ["It works"],
                    "depends_on": [],
                    "estimated_complexity": "medium",
                    "intention_mapping": {
                        "fulfills_requirements": [
                            "Create tasks with title and description",
                            "List tasks with filtering",
                            "Mark tasks as complete",
                        ],
                        "serves_purpose": True,
                        "serves_core_value": True,
                        "success_criteria_addressed": [
                            "User can create and complete tasks in under 5 seconds",
                            "All data persists between sessions",
                        ],
                    },
                }
            ]
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(milestones_data), model="mock"
        )

        gen = MilestoneGenerator()
        milestones = await gen.generate(
            [sample_plan], sample_consensus, sample_intention, mock_provider
        )

        assert len(milestones) == 1
        assert milestones[0].id == "M1"
        assert isinstance(milestones[0], Milestone)

    def test_validate_coverage_all_mapped(self, sample_intention):
        milestone = Milestone(
            id="M1",
            name="All",
            description="Everything",
            intention_mapping=IntentionMapping(
                fulfills_requirements=[
                    "Create tasks with title and description",
                    "List tasks with filtering",
                    "Mark tasks as complete",
                ],
                success_criteria_addressed=[
                    "User can create and complete tasks in under 5 seconds",
                    "All data persists between sessions",
                ],
            ),
        )

        gaps = MilestoneGenerator.validate_coverage([milestone], sample_intention)
        assert gaps == []

    def test_validate_coverage_missing_requirement(self, sample_intention):
        milestone = Milestone(
            id="M1",
            name="Partial",
            description="Only some",
            intention_mapping=IntentionMapping(
                fulfills_requirements=[
                    "Create tasks with title and description",
                    # Missing: "List tasks with filtering"
                    # Missing: "Mark tasks as complete"
                ],
                success_criteria_addressed=[
                    "User can create and complete tasks in under 5 seconds",
                    "All data persists between sessions",
                ],
            ),
        )

        gaps = MilestoneGenerator.validate_coverage([milestone], sample_intention)
        assert len(gaps) == 2
        assert any("List tasks with filtering" in g for g in gaps)
        assert any("Mark tasks as complete" in g for g in gaps)
