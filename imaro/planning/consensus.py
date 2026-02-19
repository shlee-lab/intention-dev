"""Consensus evaluation across multiple plans."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.providers.base import LLMProvider

from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import ConsensusResult, IntentionDocument, Plan
from imaro.planning.plan_generator import extract_json

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_CONSENSUS = """\
You are an expert evaluator assessing multiple implementation plans against a \
project intention.

## Project Intention
{intention_context}

## Plans to Evaluate
{plans_json}

## Instructions

Evaluate these plans on two axes:
1. **Plan Consensus** (0.0–1.0): How much do the plans agree on architecture, \
tech stack, milestone structure, and approach?
2. **Intention Alignment** (0.0–1.0): How well do the plans collectively serve \
the project intention? Are all requirements covered? Is the core value preserved?

Respond with ONLY a JSON object (no markdown fences):

{{
  "plan_consensus_score": 0.0,
  "intention_alignment_score": 0.0,
  "aligned_points": ["areas where plans agree"],
  "divergent_points": ["areas where plans disagree"],
  "intention_gaps": ["intention items not adequately addressed by any plan"]
}}

Be precise with scores. 1.0 means perfect agreement/alignment.\
"""


class ConsensusEvaluator:
    """Evaluate consensus across plans and alignment with intention."""

    async def evaluate(
        self,
        plans: list[Plan],
        intention: IntentionDocument,
        provider: LLMProvider,
        threshold: float = 0.75,
    ) -> ConsensusResult:
        intention_ctx = IntentionDocumentManager.to_prompt_context(intention)
        plans_summary = []
        for i, plan in enumerate(plans, 1):
            plans_summary.append(
                f"### Plan {i}: {plan.name}\n"
                f"Description: {plan.description}\n"
                f"Architecture: {plan.architecture_notes}\n"
                f"Tech Stack: {', '.join(plan.tech_stack)}\n"
                f"Milestones: {len(plan.milestones)}\n"
                + "\n".join(
                    f"  - {m.id}: {m.name} ({m.estimated_complexity})"
                    for m in plan.milestones
                )
            )

        prompt = PROMPT_TEMPLATE_CONSENSUS.format(
            intention_context=intention_ctx,
            plans_json="\n\n".join(plans_summary),
        )

        resp = await provider.generate(prompt, temperature=0.3, max_tokens=4096)
        try:
            data = extract_json(resp.content)
        except ValueError:
            logger.warning("Consensus JSON parse failed, retrying")
            fix_prompt = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY valid JSON.\n\n"
                f"Previous response:\n{resp.content}"
            )
            resp = await provider.generate(fix_prompt, temperature=0.2, max_tokens=4096)
            data = extract_json(resp.content)

        pc = float(data.get("plan_consensus_score", 0))
        ia = float(data.get("intention_alignment_score", 0))
        overall = 0.4 * pc + 0.6 * ia

        if overall >= threshold:
            recommendation = "proceed"
        elif overall >= threshold - 0.15:
            recommendation = "advise_against"
        else:
            recommendation = "strongly_advise_against"

        return ConsensusResult(
            plan_consensus_score=pc,
            intention_alignment_score=ia,
            overall_score=overall,
            aligned_points=data.get("aligned_points", []),
            divergent_points=data.get("divergent_points", []),
            intention_gaps=data.get("intention_gaps", []),
            recommendation=recommendation,
            user_options=["proceed_anyway", "revise_intention", "select_best_plan"],
        )
