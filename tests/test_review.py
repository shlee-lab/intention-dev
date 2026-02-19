"""Tests for Reviewer and ReviewGate."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from imaro.models.schemas import (
    IssueSeverity,
    LLMResponse,
    ReviewIssue,
    ReviewResult,
)
from imaro.review.review_gate import ReviewGate
from imaro.review.reviewer import Reviewer


# ── Reviewer ────────────────────────────────────────────────────────────────


class TestReviewer:
    @pytest.mark.asyncio
    async def test_returns_review_result_with_correct_role(
        self, sample_milestone, sample_intention, mock_provider
    ):
        review_data = {
            "summary": "Looks good",
            "passed": True,
            "intention_alignment_score": 0.95,
            "intention_alignment_details": "Well aligned",
            "issues": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(review_data), model="mock"
        )

        reviewer = Reviewer()
        result = await reviewer.review(
            role="functionality",
            milestone=sample_milestone,
            intention=sample_intention,
            code_diff="diff here",
            full_file_contents={"main.py": "print('hello')"},
            provider=mock_provider,
        )

        assert isinstance(result, ReviewResult)
        assert result.role == "functionality"
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_unknown_role_raises(
        self, sample_milestone, sample_intention, mock_provider
    ):
        reviewer = Reviewer()
        with pytest.raises(ValueError, match="Unknown reviewer role"):
            await reviewer.review(
                role="nonexistent_role",
                milestone=sample_milestone,
                intention=sample_intention,
                code_diff="",
                full_file_contents={},
                provider=mock_provider,
            )

    @pytest.mark.asyncio
    async def test_intention_alignment_role_includes_intention_context(
        self, sample_milestone, sample_intention, mock_provider
    ):
        review_data = {
            "summary": "Aligned",
            "passed": True,
            "intention_alignment_score": 0.9,
            "intention_alignment_details": "",
            "issues": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(review_data), model="mock"
        )

        reviewer = Reviewer()
        await reviewer.review(
            role="intention_alignment",
            milestone=sample_milestone,
            intention=sample_intention,
            code_diff="diff",
            full_file_contents={},
            provider=mock_provider,
        )

        # Check that the prompt included intention context
        call_args = mock_provider.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "PROJECT INTENTION" in prompt

    @pytest.mark.asyncio
    async def test_non_intention_role_omits_intention_context(
        self, sample_milestone, sample_intention, mock_provider
    ):
        review_data = {
            "summary": "OK",
            "passed": True,
            "intention_alignment_score": 1.0,
            "intention_alignment_details": "",
            "issues": [],
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(review_data), model="mock"
        )

        reviewer = Reviewer()
        await reviewer.review(
            role="functionality",
            milestone=sample_milestone,
            intention=sample_intention,
            code_diff="diff",
            full_file_contents={},
            provider=mock_provider,
        )

        call_args = mock_provider.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "PROJECT INTENTION" not in prompt


# ── ReviewGate ──────────────────────────────────────────────────────────────


class TestReviewGate:
    @pytest.mark.asyncio
    async def test_compile_report_classifies_by_severity(self):
        reviews = [
            ReviewResult(
                role="functionality",
                issues=[
                    ReviewIssue(description="Critical bug", severity=IssueSeverity.CRITICAL),
                    ReviewIssue(description="Naming", severity=IssueSeverity.NITPICK),
                ],
                summary="Some issues",
                passed=False,
            ),
            ReviewResult(
                role="edge_cases",
                issues=[
                    ReviewIssue(description="Missing check", severity=IssueSeverity.IMPROVEMENT),
                ],
                summary="Minor issues",
                passed=True,
            ),
        ]

        gate = ReviewGate()
        report = await gate.compile_report(reviews)

        assert len(report.critical_issues) == 1
        assert len(report.improvement_issues) == 1
        assert len(report.nitpick_issues) == 1

    @pytest.mark.asyncio
    async def test_no_critical_issues_approve(self):
        reviews = [
            ReviewResult(
                role="functionality",
                issues=[
                    ReviewIssue(description="Style nit", severity=IssueSeverity.NITPICK),
                ],
                summary="Looks good",
                passed=True,
            ),
        ]

        gate = ReviewGate()
        report = await gate.compile_report(reviews)

        assert report.system_recommendation == "approve"

    @pytest.mark.asyncio
    async def test_has_critical_issues_request_changes(self):
        reviews = [
            ReviewResult(
                role="functionality",
                issues=[
                    ReviewIssue(description="Security flaw", severity=IssueSeverity.CRITICAL),
                ],
                summary="Critical issues",
                passed=False,
            ),
        ]

        gate = ReviewGate()
        report = await gate.compile_report(reviews)

        assert report.system_recommendation == "request_changes"

    @pytest.mark.asyncio
    async def test_has_contradictions_discuss(self, mock_provider):
        """When no critical issues but contradictions exist → discuss."""
        reviews = [
            ReviewResult(
                role="functionality",
                issues=[],
                summary="All good",
                passed=True,
            ),
            ReviewResult(
                role="edge_cases",
                issues=[],
                summary="All good too",
                passed=True,
            ),
        ]

        contradiction_data = {
            "contradictions": [
                {
                    "reviewer_a": "functionality",
                    "reviewer_b": "edge_cases",
                    "description": "Disagree on error handling approach",
                    "resolution_hint": "Discuss with team",
                }
            ]
        }
        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps(contradiction_data), model="mock"
        )

        gate = ReviewGate()
        report = await gate.compile_report(reviews, provider=mock_provider)

        assert report.system_recommendation == "discuss"
        assert len(report.contradictions) == 1

    @pytest.mark.asyncio
    async def test_detect_contradictions_no_conflicts(self, mock_provider):
        reviews = [
            ReviewResult(role="functionality", issues=[], summary="Good", passed=True),
            ReviewResult(role="edge_cases", issues=[], summary="Good", passed=True),
        ]

        mock_provider.generate.return_value = LLMResponse(
            content=json.dumps({"contradictions": []}), model="mock"
        )

        gate = ReviewGate()
        contradictions = await gate.detect_contradictions(reviews, mock_provider)

        assert contradictions == []
