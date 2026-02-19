"""Milestone generation from consensus-evaluated plans."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.providers.base import LLMProvider

from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import (
    ConsensusResult,
    IntentionDocument,
    Milestone,
    Plan,
)
from imaro.planning.plan_generator import extract_json

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_MILESTONES = """\
You are a project planner merging the best elements of multiple reviewed plans \
into a single set of implementation milestones.

## Project Intention
{intention_context}

## Consensus Evaluation
Aligned Points: {aligned_points}
Divergent Points: {divergent_points}
Intention Gaps: {intention_gaps}

## Plans
{plans_json}

## Instructions

Merge the best elements from all plans into a unified milestone sequence. \
Prioritize aligned points and resolve divergent points by choosing the approach \
that best serves the project intention.

Milestone design rules:
- Each milestone must deliver working, testable functionality — NOT just scaffolding.
- Do NOT create separate milestones for testing, error handling, robustness, \
or documentation. Include these as part of the implementation milestones.
- Group closely related features into a single milestone (e.g. CRUD operations = 1 milestone).
- Prefer fewer, larger milestones over many small ones.

IMPORTANT: Every functional requirement and success criterion from the intention \
MUST be mapped to at least one milestone's intention_mapping.

Respond with ONLY a JSON object (no markdown fences):

{{
  "milestones": [
    {{
      "id": "M1",
      "name": "Milestone name",
      "description": "What this delivers",
      "scope": ["included items"],
      "constraints": ["limitations"],
      "acceptance_criteria": ["verifiable criteria"],
      "depends_on": [],
      "estimated_complexity": "low|medium|high",
      "intention_mapping": {{
        "fulfills_requirements": ["requirement text from intention"],
        "serves_purpose": true,
        "serves_core_value": true,
        "success_criteria_addressed": ["criterion text from intention"]
      }}
    }}
  ]
}}\
"""


class MilestoneGenerator:
    """Generate milestones from consensus-evaluated plans."""

    async def generate(
        self,
        plans: list[Plan],
        consensus: ConsensusResult,
        intention: IntentionDocument,
        provider: LLMProvider,
        max_milestones: int = 10,
    ) -> list[Milestone]:
        intention_ctx = IntentionDocumentManager.to_prompt_context(intention)

        plans_text = []
        for i, plan in enumerate(plans, 1):
            milestones_text = "\n".join(
                f"    {m.id}: {m.name} — {m.description} "
                f"(complexity: {m.estimated_complexity})"
                for m in plan.milestones
            )
            plans_text.append(
                f"### Plan {i}: {plan.name}\n"
                f"  Architecture: {plan.architecture_notes}\n"
                f"  Tech Stack: {', '.join(plan.tech_stack)}\n"
                f"  Milestones:\n{milestones_text}"
            )

        prompt = PROMPT_TEMPLATE_MILESTONES.format(
            intention_context=intention_ctx,
            aligned_points="\n".join(f"  - {p}" for p in consensus.aligned_points),
            divergent_points="\n".join(f"  - {p}" for p in consensus.divergent_points),
            intention_gaps="\n".join(f"  - {g}" for g in consensus.intention_gaps),
            plans_json="\n\n".join(plans_text),
        )

        milestones = await self._generate_and_parse(provider, prompt)

        # If over limit, ask LLM to consolidate
        if len(milestones) > max_milestones:
            logger.info(
                "Generated %d milestones, exceeds max %d — requesting consolidation",
                len(milestones), max_milestones,
            )
            milestones = await self._consolidate(
                provider, milestones, max_milestones, intention_ctx
            )

        gaps = self.validate_coverage(milestones, intention)
        if gaps:
            logger.warning("Unmapped intention items: %s", gaps)

        return milestones

    async def _generate_and_parse(
        self, provider: LLMProvider, prompt: str
    ) -> list[Milestone]:
        """Call LLM, parse JSON, return milestones."""
        resp = await provider.generate(prompt, temperature=0.4, max_tokens=8192)
        try:
            data = extract_json(resp.content)
        except ValueError:
            logger.warning("Milestone JSON parse failed, retrying")
            fix_prompt = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY valid JSON.\n\n"
                f"Previous response:\n{resp.content}"
            )
            resp = await provider.generate(fix_prompt, temperature=0.2, max_tokens=8192)
            data = extract_json(resp.content)

        milestones_data = data.get("milestones", data) if isinstance(data, dict) else data
        return [Milestone.model_validate(m) for m in milestones_data]

    async def _consolidate(
        self,
        provider: LLMProvider,
        milestones: list[Milestone],
        max_milestones: int,
        intention_ctx: str,
    ) -> list[Milestone]:
        """Ask LLM to merge milestones down to the limit."""
        ms_text = "\n".join(
            f"  {m.id}: {m.name} — {m.description}" for m in milestones
        )
        prompt = (
            f"You produced {len(milestones)} milestones but the maximum allowed "
            f"is {max_milestones}.\n\n"
            f"## Current Milestones\n{ms_text}\n\n"
            f"## Project Intention\n{intention_ctx}\n\n"
            f"Merge related milestones to produce at most {max_milestones} milestones. "
            "Keep the same JSON format. Ensure all requirements remain covered.\n\n"
            "Respond with ONLY a JSON object: {\"milestones\": [...]}"
        )
        return await self._generate_and_parse(provider, prompt)

    @staticmethod
    def validate_coverage(
        milestones: list[Milestone], intention: IntentionDocument
    ) -> list[str]:
        """Return intention items not mapped to any milestone."""
        mapped_reqs: set[str] = set()
        mapped_criteria: set[str] = set()
        for ms in milestones:
            mapped_reqs.update(ms.intention_mapping.fulfills_requirements)
            mapped_criteria.update(ms.intention_mapping.success_criteria_addressed)

        gaps: list[str] = []
        for req in intention.functional_requirements:
            if req not in mapped_reqs:
                gaps.append(f"Unmapped requirement: {req}")
        for crit in intention.success_criteria:
            if crit not in mapped_criteria:
                gaps.append(f"Unmapped success criterion: {crit}")
        return gaps
