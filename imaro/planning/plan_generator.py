"""Multi-agent plan generation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.providers.base import LLMProvider

from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import IntentionDocument, Plan

logger = logging.getLogger(__name__)

AGENT_PERSPECTIVES = [
    (
        "simplicity",
        "You prioritize simplicity and minimalism. Prefer fewer moving parts, "
        "straightforward architecture, and the smallest viable solution.",
    ),
    (
        "robustness",
        "You prioritize robustness and reliability. Emphasize error handling, "
        "validation, testing strategy, and graceful degradation.",
    ),
    (
        "extensibility",
        "You prioritize extensibility and clean architecture. Emphasize clear "
        "separation of concerns, well-defined interfaces, and future adaptability.",
    ),
]

PROMPT_TEMPLATE_PLAN = """\
You are a software architect with a focus on {perspective_name}.

{perspective_desc}

## Project Intention
{intention_context}

## Instructions

Create a detailed implementation plan for this project. Return ONLY a JSON object \
(no markdown fences) with this structure:

{{
  "name": "Plan name reflecting your perspective",
  "description": "Brief plan summary",
  "architecture_notes": "Key architectural decisions",
  "tech_stack": ["tech1", "tech2"],
  "intention_alignment": {{
    "how_purpose_served": "...",
    "how_core_value_delivered": "...",
    "requirements_coverage": "..."
  }},
  "milestones": [
    {{
      "id": "M1",
      "name": "Milestone name",
      "description": "What this milestone delivers",
      "scope": ["what is included"],
      "constraints": ["limitations"],
      "acceptance_criteria": ["verifiable criteria"],
      "depends_on": [],
      "estimated_complexity": "low|medium|high",
      "intention_mapping": {{
        "fulfills_requirements": ["req from intention"],
        "serves_purpose": true,
        "serves_core_value": true,
        "success_criteria_addressed": ["criteria from intention"]
      }}
    }}
  ]
}}

Ensure every functional requirement and success criterion from the intention is \
addressed by at least one milestone.

Milestone granularity rules:
- Group related features into a single milestone (e.g. CRUD operations = 1 milestone, \
not 4 separate ones).
- Each milestone must deliver a meaningful, independently testable unit of functionality.
- Prefer fewer, larger milestones over many small ones.\
"""


def extract_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


class PlanGenerator:
    """Generate multiple plans in parallel using different agent perspectives."""

    async def generate_plans(
        self,
        intention: IntentionDocument,
        num_agents: int = 3,
        providers: list[LLMProvider] | None = None,
        default_provider: LLMProvider | None = None,
    ) -> list[Plan]:
        if providers is None:
            if default_provider is None:
                raise ValueError("Either providers or default_provider is required")
            providers = [default_provider] * num_agents

        intention_ctx = IntentionDocumentManager.to_prompt_context(intention)
        tasks = []
        for i in range(min(num_agents, len(AGENT_PERSPECTIVES))):
            name, desc = AGENT_PERSPECTIVES[i]
            prompt = PROMPT_TEMPLATE_PLAN.format(
                perspective_name=name,
                perspective_desc=desc,
                intention_context=intention_ctx,
            )
            tasks.append(self._generate_one(providers[i], prompt, name))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        plans: list[Plan] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Agent %d failed: %s", i, result)
            else:
                plans.append(result)

        if not plans:
            raise RuntimeError("All plan generation agents failed")
        return plans

    async def _generate_one(
        self, provider: LLMProvider, prompt: str, perspective: str
    ) -> Plan:
        logger.debug("Generating plan with %s perspective", perspective)
        resp = await provider.generate(prompt, temperature=0.7, max_tokens=8192)
        try:
            data = extract_json(resp.content)
        except ValueError:
            logger.warning("JSON parse failed for %s, retrying", perspective)
            fix_prompt = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY valid JSON.\n\n"
                f"Previous response:\n{resp.content}"
            )
            resp = await provider.generate(fix_prompt, temperature=0.3, max_tokens=8192)
            data = extract_json(resp.content)

        return Plan.model_validate(data)
