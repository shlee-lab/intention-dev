"""Main orchestrator — wires all IMARO modules together."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from imaro.config import IMAROConfig
from imaro.execution.claude_code import ClaudeCodeExecutor
from imaro.execution.context_manager import ContextManager
from imaro.intention.document import IntentionDocumentManager
from imaro.intention.refiner import IntentionRefiner
from imaro.models.schemas import (
    DriftReport,
    IntentionDocument,
    MilestoneResult,
    MilestoneStatus,
    Plan,
    ProjectResult,
    RefinementRound,
    UserReviewDecision,
)
from imaro.planning.consensus import ConsensusEvaluator
from imaro.planning.milestone import MilestoneGenerator
from imaro.planning.plan_generator import PlanGenerator
from imaro.review.review_gate import ReviewGate
from imaro.review.reviewer import Reviewer
from imaro.review.roles import REVIEWER_ROLES
from imaro.ui.terminal import TerminalUI

logger = logging.getLogger(__name__)

IMARO_DIR = ".imaro"


class Orchestrator:
    """Run the full IMARO pipeline end-to-end."""

    def __init__(
        self,
        config: IMAROConfig | None = None,
        ui: TerminalUI | None = None,
    ) -> None:
        self.config = config or IMAROConfig()
        self.ui = ui or TerminalUI()

        self.doc_manager = IntentionDocumentManager()
        self.refiner = IntentionRefiner()
        self.plan_generator = PlanGenerator()
        self.consensus_evaluator = ConsensusEvaluator()
        self.milestone_generator = MilestoneGenerator()
        self.executor = ClaudeCodeExecutor(
            allowed_tools=self.config.claude_code_allowed_tools,
        )
        self.context_manager = ContextManager()
        self.reviewer = Reviewer()
        self.review_gate = ReviewGate()

    async def run(
        self,
        raw_input: str,
        project_path: str | Path,
    ) -> ProjectResult:
        project_path = Path(project_path).resolve()
        project_path.mkdir(parents=True, exist_ok=True)
        state_dir = project_path / IMARO_DIR
        state_dir.mkdir(parents=True, exist_ok=True)

        result = ProjectResult()

        try:
            # ── Phase 1: Intention Refinement ───────────────────────────
            self.ui.show_progress("Intention Refinement", "Analyzing project description")
            intention = await self._refine_intention(raw_input)
            result.intention = intention
            self.doc_manager.save(intention, project_path)
            self._save_state(state_dir, "intention", intention.model_dump())

            # ── Phase 2: Plan Generation ────────────────────────────────
            self.ui.show_progress("Plan Generation", f"Generating {self.config.plan_agents} plans")
            plans = await self._generate_plans(intention)
            result.plans = plans
            self._save_state(state_dir, "plans", [p.model_dump() for p in plans])

            # ── Phase 3: Consensus Evaluation ───────────────────────────
            self.ui.show_progress("Consensus Evaluation")
            consensus_provider = self.config.get_provider("consensus")
            consensus = await self.consensus_evaluator.evaluate(
                plans, intention, consensus_provider, self.config.consensus_threshold
            )
            result.consensus = consensus
            self._save_state(state_dir, "consensus", consensus.model_dump())

            self.ui.display_consensus(consensus)

            if consensus.recommendation != "proceed":
                plans, intention = await self._handle_consensus_failure(
                    consensus, plans, intention, project_path
                )
                result.plans = plans
                result.intention = intention

            # ── Phase 4: Milestone Generation ───────────────────────────
            self.ui.show_progress("Milestone Generation")
            ms_provider = self.config.get_provider("milestone_generator")
            milestones = await self.milestone_generator.generate(
                plans, consensus, intention, ms_provider
            )
            result.milestones = milestones
            self._save_state(
                state_dir, "milestones", [m.model_dump() for m in milestones]
            )

            approved = await asyncio.to_thread(
                self.ui.confirm_milestones, milestones, intention
            )
            if not approved:
                self.ui.show_warning("Milestones not approved. Aborting.")
                result.error = "Milestones not approved by user"
                return result

            # ── Phase 5: Execute & Review Each Milestone ────────────────
            completed_results: list[MilestoneResult] = []
            for milestone in milestones:
                self.ui.show_progress(
                    "Milestone Execution",
                    f"{milestone.id}: {milestone.name}",
                )

                ms_result = await self._execute_and_review_milestone(
                    milestone, intention, project_path, completed_results
                )
                completed_results.append(ms_result)
                result.milestone_results.append(ms_result)
                self._save_state(
                    state_dir,
                    f"milestone_{milestone.id}",
                    ms_result.model_dump(),
                )

                if ms_result.status == MilestoneStatus.FAILED:
                    self.ui.show_error(
                        f"Milestone {milestone.id} failed. Stopping."
                    )
                    result.error = f"Milestone {milestone.id} failed"
                    return result

                # Drift check after each milestone
                await self._check_drift(
                    completed_results, intention, project_path
                )

            self.context_manager.restore_context(project_path)
            result.success = True
            self.ui.show_success("All milestones completed successfully!")

        except KeyboardInterrupt:
            self.ui.show_warning("Interrupted by user")
            result.error = "Interrupted"
        except Exception as exc:
            logger.exception("Orchestrator error")
            self.ui.show_error(f"Error: {exc}")
            result.error = str(exc)

        self._save_state(state_dir, "result", result.model_dump())
        return result

    # ── Intention ────────────────────────────────────────────────────────

    async def _refine_intention(self, raw_input: str) -> IntentionDocument:
        provider = self.config.get_provider("refiner")
        history: list[RefinementRound] = []

        for round_num in range(1, self.config.max_refinement_rounds + 1):
            if not history:
                result, ready = await self.refiner.refine(
                    raw_input, provider, self.config.max_refinement_rounds
                )
            else:
                result, ready = await self.refiner.refine_with_answers(
                    raw_input, history, provider
                )

            if ready:
                assert isinstance(result, IntentionDocument)
                approved = await asyncio.to_thread(
                    self.ui.confirm_intention, result
                )
                if approved:
                    return result
                result = await asyncio.to_thread(self.ui.edit_intention, result)
                approved = await asyncio.to_thread(
                    self.ui.confirm_intention, result
                )
                if approved:
                    return result
                # If still not approved, continue refining
                continue

            # result is list[RefinementQuestion]
            answers = await asyncio.to_thread(
                self.ui.ask_refinement_questions, result
            )
            skip = await asyncio.to_thread(self.ui.ask_skip_refinement)
            history.append(RefinementRound(questions=result, answers=answers))
            if skip:
                break

        # Final attempt after all rounds
        if history:
            result, ready = await self.refiner.refine_with_answers(
                raw_input, history, provider
            )
            if ready and isinstance(result, IntentionDocument):
                approved = await asyncio.to_thread(
                    self.ui.confirm_intention, result
                )
                if approved:
                    return result

        raise RuntimeError("Could not produce an approved intention document")

    # ── Planning ─────────────────────────────────────────────────────────

    async def _generate_plans(self, intention: IntentionDocument) -> list[Plan]:
        provider = self.config.get_provider("planner")
        return await self.plan_generator.generate_plans(
            intention,
            num_agents=self.config.plan_agents,
            default_provider=provider,
        )

    async def _handle_consensus_failure(
        self,
        consensus,
        plans: list[Plan],
        intention: IntentionDocument,
        project_path: Path,
    ) -> tuple[list[Plan], IntentionDocument]:
        choice = await asyncio.to_thread(
            self.ui.handle_consensus_failure, consensus, plans, intention
        )

        if choice == "proceed_anyway":
            return plans, intention

        if choice == "select_best_plan":
            idx = await asyncio.to_thread(self.ui.select_plan, plans)
            return [plans[idx]], intention

        if choice == "revise_intention":
            intention = await asyncio.to_thread(
                self.ui.edit_intention, intention
            )
            self.doc_manager.save(intention, project_path)
            plans = await self._generate_plans(intention)
            return plans, intention

        if choice == "regenerate_plans":
            plans = await self._generate_plans(intention)
            return plans, intention

        raise RuntimeError("User chose to abort")

    # ── Execution & Review ───────────────────────────────────────────────

    async def _execute_and_review_milestone(
        self,
        milestone,
        intention: IntentionDocument,
        project_path: Path,
        previous: list[MilestoneResult],
    ) -> MilestoneResult:
        ms_result = MilestoneResult(
            milestone=milestone, status=MilestoneStatus.EXECUTING
        )

        # Write CLAUDE.md context
        self.context_manager.write_context(
            milestone, intention, project_path, previous
        )

        # Execute
        exec_result = await self.executor.execute_milestone(
            milestone, intention, project_path
        )
        ms_result.execution = exec_result

        if not exec_result.success:
            self.ui.show_error(
                f"Execution failed: {exec_result.error}"
            )
            ms_result.status = MilestoneStatus.FAILED
            return ms_result

        self.ui.show_success(f"Milestone {milestone.id} executed")

        # Review loop
        for attempt in range(1, self.config.max_fix_attempts + 1):
            ms_result.status = MilestoneStatus.REVIEWING
            self.ui.show_progress("Review", f"Attempt {attempt}")

            # Get diff and artifacts
            diff = await self.executor.get_diff(project_path)
            artifacts = await self.executor.get_changed_artifacts(project_path)

            # Run all reviewers in parallel
            review_provider = self.config.get_provider("reviewer")
            review_tasks = [
                self.reviewer.review(
                    role, milestone, intention, diff, artifacts, review_provider
                )
                for role in REVIEWER_ROLES
            ]
            reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
            from imaro.models.schemas import ReviewResult

            valid_reviews = []
            for r in reviews:
                if isinstance(r, ReviewResult):
                    valid_reviews.append(r)
                elif isinstance(r, Exception):
                    logger.error("Reviewer failed: %s", r)

            ms_result.reviews = valid_reviews

            # Compile report
            contradiction_provider = self.config.get_provider("contradiction_detector")
            report = await self.review_gate.compile_report(
                valid_reviews, contradiction_provider
            )
            ms_result.review_report = report

            # Present to user
            decision = await asyncio.to_thread(
                self.ui.present_review_report, milestone, report, intention
            )

            if decision == UserReviewDecision.APPROVE:
                ms_result.status = MilestoneStatus.COMPLETED
                return ms_result

            if decision == UserReviewDecision.SKIP:
                ms_result.status = MilestoneStatus.COMPLETED
                self.ui.show_warning("Skipping review, accepting as-is")
                return ms_result

            if decision == UserReviewDecision.ABORT:
                ms_result.status = MilestoneStatus.FAILED
                return ms_result

            # FIX — build fix instructions from critical issues
            fix_lines = ["Fix the following issues found in code review:"]
            for issue in report.critical_issues:
                fix_lines.append(f"- [{issue.severity.value}] {issue.description}")
                if issue.suggestion:
                    fix_lines.append(f"  Suggestion: {issue.suggestion}")
            for issue in report.improvement_issues:
                fix_lines.append(f"- [{issue.severity.value}] {issue.description}")

            fix_instructions = "\n".join(fix_lines)
            session_id = exec_result.session_id

            exec_result = await self.executor.fix_issues(
                fix_instructions, session_id, milestone, intention, project_path
            )
            ms_result.execution = exec_result

            if not exec_result.success:
                self.ui.show_error(f"Fix attempt {attempt} failed")
                ms_result.status = MilestoneStatus.FAILED
                return ms_result

        self.ui.show_warning("Max fix attempts reached")
        ms_result.status = MilestoneStatus.COMPLETED
        return ms_result

    # ── Drift Detection ──────────────────────────────────────────────────

    async def _check_drift(
        self,
        completed: list[MilestoneResult],
        intention: IntentionDocument,
        project_path: Path,
    ) -> None:
        if len(completed) < 2:
            return  # Skip drift check for first milestone

        provider = self.config.get_provider("drift_detector")
        intention_ctx = IntentionDocumentManager.to_prompt_context(intention)

        milestones_summary = "\n".join(
            f"- {mr.milestone.id}: {mr.milestone.name} — {mr.status.value}"
            for mr in completed
            if mr.milestone
        )

        prompt = (
            "Evaluate whether the project is drifting from its original intention.\n\n"
            f"## Intention\n{intention_ctx}\n\n"
            f"## Completed Milestones\n{milestones_summary}\n\n"
            "Respond with JSON: "
            '{"drift_score": 0.0, "drifted_areas": [], '
            '"recommendations": [], "still_aligned": true}\n'
            "drift_score 0.0 = no drift, 1.0 = completely off track."
        )

        resp = await provider.generate(prompt, temperature=0.3, max_tokens=2048)
        try:
            from imaro.planning.plan_generator import extract_json

            data = extract_json(resp.content)
            drift = DriftReport.model_validate(data)
        except (ValueError, Exception) as exc:
            logger.warning("Drift detection failed: %s", exc)
            return

        if drift.drift_score >= self.config.drift_alert_threshold:
            choice = await asyncio.to_thread(
                self.ui.handle_drift, drift, intention
            )
            if choice == "abort":
                raise RuntimeError("User aborted due to drift")

    # ── State Persistence ────────────────────────────────────────────────

    @staticmethod
    def _save_state(state_dir: Path, name: str, data: dict | list) -> None:
        path = state_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
