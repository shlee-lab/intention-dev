"""Intention refinement via LLM-driven clarifying questions."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.models.schemas import LLMResponse
    from imaro.providers.base import LLMProvider

from imaro.models.schemas import (
    IntentionDocument,
    RefinementQuestion,
    RefinementRound,
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_REFINE = """\
You are an expert requirements analyst. Analyze the following project description \
and either ask clarifying questions or produce a final structured intention document.

## Project Description
{raw_input}

{history_section}

## Instructions

If you need more information to produce a high-quality intention document, respond with \
clarifying questions. If you have enough information, produce the final document.

Respond with ONLY a JSON object (no markdown fences) in one of these two formats:

### Format A — Need more info:
{{
  "status": "questions",
  "questions": [
    {{"question": "...", "context": "why this matters"}}
  ]
}}

### Format B — Ready to produce document:
{{
  "status": "ready",
  "document": {{
    "purpose": "one-sentence purpose",
    "target_users": "who will use this",
    "core_value": "the single most important value proposition",
    "functional_requirements": ["req1", "req2"],
    "non_functional_requirements": ["nfr1"],
    "out_of_scope": ["excluded1"],
    "constraints": ["constraint1"],
    "success_criteria": ["criterion1"]
  }}
}}

Generate 3-5 focused questions if needed. Be specific and actionable.\
"""


def _extract_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling code fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


async def _generate_with_retry(
    provider: LLMProvider, prompt: str, system: str = ""
) -> dict:
    """Call LLM and parse JSON, retrying once on parse failure."""
    resp: LLMResponse = await provider.generate(
        prompt, system=system, temperature=0.5, max_tokens=4096
    )
    try:
        return _extract_json(resp.content)
    except ValueError:
        logger.warning("JSON parse failed, retrying with fix prompt")
        fix_prompt = (
            f"Your previous response was not valid JSON. "
            f"Please fix it and respond with ONLY valid JSON.\n\n"
            f"Previous response:\n{resp.content}"
        )
        resp = await provider.generate(fix_prompt, system=system, temperature=0.3)
        return _extract_json(resp.content)


class IntentionRefiner:
    """Refine a raw project description into a structured IntentionDocument."""

    async def refine(
        self,
        raw_input: str,
        provider: LLMProvider,
        max_rounds: int = 3,
    ) -> tuple[IntentionDocument | list[RefinementQuestion], bool]:
        """Start refinement from a raw description.

        Returns:
            (IntentionDocument, True) if the LLM produced a document directly.
            (list[RefinementQuestion], False) if clarifying questions are needed.
        """
        prompt = PROMPT_TEMPLATE_REFINE.format(
            raw_input=raw_input,
            history_section="",
        )

        data = await _generate_with_retry(provider, prompt)
        return self._parse_response(data, raw_input)

    async def refine_with_answers(
        self,
        raw_input: str,
        history: list[RefinementRound],
        provider: LLMProvider,
    ) -> tuple[IntentionDocument | list[RefinementQuestion], bool]:
        """Continue refinement with accumulated Q&A history.

        Returns:
            (IntentionDocument, True) if ready.
            (list[RefinementQuestion], False) if more questions needed.
        """
        history_lines = ["## Previous Q&A Rounds"]
        for i, rnd in enumerate(history, 1):
            history_lines.append(f"\n### Round {i}")
            for q, a in zip(rnd.questions, rnd.answers):
                history_lines.append(f"Q: {q.question}")
                history_lines.append(f"A: {a}")

        history_lines.append(
            "\n## IMPORTANT\n"
            "You have received answers to your clarifying questions. You now have "
            "sufficient information. Produce the final intention document (Format B). "
            "Do NOT ask more questions."
        )

        prompt = PROMPT_TEMPLATE_REFINE.format(
            raw_input=raw_input,
            history_section="\n".join(history_lines),
        )

        data = await _generate_with_retry(provider, prompt)
        return self._parse_response(data, raw_input)

    @staticmethod
    def _parse_response(
        data: dict, raw_input: str
    ) -> tuple[IntentionDocument | list[RefinementQuestion], bool]:
        status = data.get("status", "")

        if status == "ready":
            doc_data = data.get("document", data)
            doc_data["raw_input"] = raw_input
            doc = IntentionDocument.model_validate(doc_data)
            return doc, True

        if status == "questions":
            questions = [
                RefinementQuestion.model_validate(q)
                for q in data.get("questions", [])
            ]
            return questions, False

        raise ValueError(f"Unexpected refinement status: {status!r}")
