"""Individual milestone reviewer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.providers.base import LLMProvider

from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import (
    IntentionDocument,
    Milestone,
    ReviewResult,
)
from imaro.planning.plan_generator import extract_json
from imaro.review.roles import REVIEWER_ROLES

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_REVIEW = """\
{system_prompt}

## Milestone
Name: {milestone_name}
Description: {milestone_description}
Acceptance Criteria:
{acceptance_criteria}

{intention_section}

## Code Diff
```
{code_diff}
```

## Full File Contents
{file_contents}

## Instructions

Review the code and respond with ONLY a JSON object (no markdown fences):

{{
  "summary": "brief review summary",
  "passed": true,
  "intention_alignment_score": 1.0,
  "intention_alignment_details": "explanation of alignment (if reviewing intention)",
  "issues": [
    {{
      "description": "issue description",
      "severity": "critical|improvement|nitpick",
      "file": "path/to/file",
      "line": null,
      "suggestion": "how to fix",
      "intention_relevant": false
    }}
  ]
}}

Only include issues you actually found. An empty issues list is fine if the code looks good.\
"""


class Reviewer:
    """Run a single reviewer role against milestone output."""

    async def review(
        self,
        role: str,
        milestone: Milestone,
        intention: IntentionDocument,
        code_diff: str,
        full_file_contents: dict[str, str],
        provider: LLMProvider,
    ) -> ReviewResult:
        role_config = REVIEWER_ROLES.get(role)
        if role_config is None:
            raise ValueError(f"Unknown reviewer role: {role}")

        intention_section = ""
        if role_config["reviews_intention"]:
            intention_section = (
                "## Project Intention\n"
                + IntentionDocumentManager.to_prompt_context(intention)
            )

        file_contents_str = ""
        for fpath, content in full_file_contents.items():
            file_contents_str += f"\n### {fpath}\n```\n{content}\n```\n"

        ac = "\n".join(f"  - {c}" for c in milestone.acceptance_criteria)

        prompt = PROMPT_TEMPLATE_REVIEW.format(
            system_prompt=role_config["system_prompt"],
            milestone_name=milestone.name,
            milestone_description=milestone.description,
            acceptance_criteria=ac,
            intention_section=intention_section,
            code_diff=code_diff[:30000],  # Truncate very large diffs
            file_contents=file_contents_str[:50000],
        )

        resp = await provider.generate(prompt, temperature=0.3, max_tokens=4096)
        try:
            data = extract_json(resp.content)
        except ValueError:
            logger.warning("Review JSON parse failed for %s, retrying", role)
            fix_prompt = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY valid JSON.\n\n"
                f"Previous response:\n{resp.content}"
            )
            resp = await provider.generate(fix_prompt, temperature=0.2, max_tokens=4096)
            data = extract_json(resp.content)

        data["role"] = role
        return ReviewResult.model_validate(data)
