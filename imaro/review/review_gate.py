"""Review gate — compile reports and detect contradictions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.providers.base import LLMProvider

from imaro.models.schemas import (
    Contradiction,
    IssueSeverity,
    ReviewReport,
    ReviewResult,
)
from imaro.planning.plan_generator import extract_json

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_CONTRADICTIONS = """\
You are analyzing multiple code reviews for the same milestone to find contradictions.

## Reviews
{reviews_text}

## Instructions

Identify any contradictions where reviewers disagree or give conflicting advice. \
A contradiction is when one reviewer says something is good while another says it's \
bad, or when their suggestions conflict.

Respond with ONLY a JSON object (no markdown fences):

{{
  "contradictions": [
    {{
      "reviewer_a": "role name",
      "reviewer_b": "role name",
      "description": "what they disagree on",
      "resolution_hint": "suggested way to resolve"
    }}
  ]
}}

Return an empty contradictions list if there are no conflicts.\
"""


class ReviewGate:
    """Compile review reports and detect contradictions."""

    async def compile_report(
        self,
        reviews: list[ReviewResult],
        provider: LLMProvider | None = None,
    ) -> ReviewReport:
        """Compile individual reviews into a unified report."""
        critical = []
        improvement = []
        nitpick = []

        for review in reviews:
            for issue in review.issues:
                if issue.severity == IssueSeverity.CRITICAL:
                    critical.append(issue)
                elif issue.severity == IssueSeverity.IMPROVEMENT:
                    improvement.append(issue)
                else:
                    nitpick.append(issue)

        contradictions = []
        if provider and len(reviews) > 1:
            contradictions = await self.detect_contradictions(reviews, provider)

        if critical:
            recommendation = "request_changes"
        elif contradictions:
            recommendation = "discuss"
        else:
            recommendation = "approve"

        return ReviewReport(
            reviews=reviews,
            contradictions=contradictions,
            critical_issues=critical,
            improvement_issues=improvement,
            nitpick_issues=nitpick,
            system_recommendation=recommendation,
        )

    async def detect_contradictions(
        self,
        reviews: list[ReviewResult],
        provider: LLMProvider,
    ) -> list[Contradiction]:
        """Use LLM to find conflicting opinions across reviews."""
        reviews_text = []
        for r in reviews:
            issues_str = "\n".join(
                f"    - [{i.severity.value}] {i.description}"
                for i in r.issues
            )
            reviews_text.append(
                f"### {r.role}\n"
                f"  Summary: {r.summary}\n"
                f"  Passed: {r.passed}\n"
                f"  Issues:\n{issues_str or '    (none)'}"
            )

        prompt = PROMPT_TEMPLATE_CONTRADICTIONS.format(
            reviews_text="\n\n".join(reviews_text)
        )

        resp = await provider.generate(prompt, temperature=0.3, max_tokens=2048)
        try:
            data = extract_json(resp.content)
        except ValueError:
            logger.warning("Contradiction detection JSON parse failed")
            return []

        return [
            Contradiction.model_validate(c)
            for c in data.get("contradictions", [])
        ]
