"""Main orchestrator — wires all IMARO modules together."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path

from imaro.config import IMAROConfig
from imaro.execution.context_manager import ContextManager
from imaro.intention.document import IntentionDocumentManager
from imaro.intention.refiner import IntentionRefiner
from imaro.models.usage import TrackedProvider, UsageTracker
from imaro.providers.base import LLMProvider
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
        interactive: bool = True,
    ) -> None:
        self.config = config or IMAROConfig()
        self.ui = ui or TerminalUI()
        self.interactive = interactive
        self.usage_tracker = UsageTracker()

        self.doc_manager = IntentionDocumentManager()
        self.refiner = IntentionRefiner()
        self.plan_generator = PlanGenerator()
        self.consensus_evaluator = ConsensusEvaluator()
        self.milestone_generator = MilestoneGenerator()
        self.executor = self.config.get_executor()
        self.context_manager = ContextManager()
        self.reviewer = Reviewer()
        self.review_gate = ReviewGate()

    def _get_provider(self, role: str) -> LLMProvider:
        """Get a provider wrapped with usage tracking."""
        provider = self.config.get_provider(role)
        return TrackedProvider(provider, self.usage_tracker, role)

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
            consensus_provider = self._get_provider("consensus")
            consensus = await self.consensus_evaluator.evaluate(
                plans, intention, consensus_provider, self.config.consensus_threshold
            )
            result.consensus = consensus
            self._save_state(state_dir, "consensus", consensus.model_dump())

            self.ui.display_consensus(consensus)

            if consensus.recommendation != "proceed":
                if self.interactive:
                    plans, intention = await self._handle_consensus_failure(
                        consensus, plans, intention, project_path
                    )
                    result.plans = plans
                    result.intention = intention
                else:
                    self.ui.show_warning(
                        f"Consensus: {consensus.recommendation} "
                        f"(score={consensus.overall_score:.2f}) — proceeding anyway"
                    )

            # ── Phase 4: Milestone Generation ───────────────────────────
            self.ui.show_progress("Milestone Generation")
            ms_provider = self._get_provider("milestone_generator")
            milestones = await self.milestone_generator.generate(
                plans, consensus, intention, ms_provider,
                max_milestones=self.config.max_milestones,
            )
            result.milestones = milestones
            self._save_state(
                state_dir, "milestones", [m.model_dump() for m in milestones]
            )

            if self.interactive:
                approved = await asyncio.to_thread(
                    self.ui.confirm_milestones, milestones, intention
                )
                if not approved:
                    self.ui.show_warning("Milestones not approved. Aborting.")
                    result.error = "Milestones not approved by user"
                    return result
            else:
                self.ui.show_progress(
                    "Milestones", f"Auto-approved {len(milestones)} milestones"
                )

            # ── Phase 5: Execute & Review Each Milestone ────────────────
            self._ensure_git_repo(project_path)

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

            if self.config.executor_type == "claude":
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
        finally:
            # Always attach and display usage summary
            summary = self.usage_tracker.summary()
            result.usage_summary = {
                "total_input_tokens": summary.total_input_tokens,
                "total_output_tokens": summary.total_output_tokens,
                "total_cost": round(summary.total_cost, 4),
                "total_calls": summary.total_calls,
                "by_role": [
                    {
                        "role": rb.role,
                        "calls": rb.calls,
                        "input_tokens": rb.input_tokens,
                        "output_tokens": rb.output_tokens,
                        "cost": round(rb.cost, 4),
                    }
                    for rb in summary.by_role
                ],
            }
            self._display_usage_summary(summary)

        self._save_state(state_dir, "result", result.model_dump())
        return result

    # ── Intention ────────────────────────────────────────────────────────

    async def _refine_intention(self, raw_input: str) -> IntentionDocument:
        provider = self._get_provider("refiner")

        # Non-interactive: skip Q&A, tell refiner to produce document directly
        if not self.interactive:
            self.ui.show_progress("Intention Refinement", "Non-interactive mode — skipping Q&A")
            result, ready = await self.refiner.refine(
                raw_input, provider, max_rounds=0
            )
            if ready and isinstance(result, IntentionDocument):
                self.ui.display_intention(result)
                return result
            # If questions came back, force a final pass with no answers
            result, ready = await self.refiner.refine_with_answers(
                raw_input, [], provider
            )
            if ready and isinstance(result, IntentionDocument):
                self.ui.display_intention(result)
                return result
            raise RuntimeError("Refiner could not produce document in non-interactive mode")

        # Interactive mode: full Q&A loop
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
        provider = self._get_provider("planner")
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

        # Write CLAUDE.md context (only needed for Claude CLI executor)
        if self.config.executor_type == "claude":
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

        # Commit so reviewers can see a meaningful diff
        self._commit_milestone(project_path, milestone.id)

        # Review loop
        for attempt in range(1, self.config.max_fix_attempts + 1):
            ms_result.status = MilestoneStatus.REVIEWING
            self.ui.show_progress("Review", f"Attempt {attempt}")

            # Get diff and artifacts
            diff = await self.executor.get_diff(project_path)
            artifacts = await self.executor.get_changed_artifacts(project_path)

            # Run all reviewers in parallel
            review_provider = self._get_provider("reviewer")
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
            contradiction_provider = self._get_provider("contradiction_detector")
            report = await self.review_gate.compile_report(
                valid_reviews, contradiction_provider
            )
            ms_result.review_report = report

            # Present to user
            if self.interactive:
                decision = await asyncio.to_thread(
                    self.ui.present_review_report, milestone, report, intention
                )
            else:
                # Auto-approve if no critical issues, auto-fix otherwise
                if not report.critical_issues:
                    decision = UserReviewDecision.APPROVE
                else:
                    decision = UserReviewDecision.FIX
                self.ui.show_progress(
                    "Review",
                    f"Auto-{decision.value} ({len(report.critical_issues)} critical issues)",
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

            # Commit fix so next review cycle sees the diff
            self._commit_milestone(project_path, f"{milestone.id}-fix{attempt}")

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

        provider = self._get_provider("drift_detector")
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
            if self.interactive:
                choice = await asyncio.to_thread(
                    self.ui.handle_drift, drift, intention
                )
                if choice == "abort":
                    raise RuntimeError("User aborted due to drift")
            else:
                self.ui.show_warning(
                    f"Drift detected (score={drift.drift_score:.2f}) — continuing anyway"
                )

    # ── Usage Display ────────────────────────────────────────────────────

    def _display_usage_summary(self, summary) -> None:
        """Print a usage/cost summary to the terminal."""
        if summary.total_calls == 0:
            return
        self.ui.show_progress("Usage Summary")
        for rb in summary.by_role:
            self.ui.show_progress(
                f"  {rb.role}",
                f"{rb.calls} calls | "
                f"{rb.input_tokens:,} in / {rb.output_tokens:,} out | "
                f"${rb.cost:.4f}",
            )
        self.ui.show_progress(
            "  Total",
            f"{summary.total_calls} calls | "
            f"{summary.total_input_tokens:,} in / {summary.total_output_tokens:,} out | "
            f"${summary.total_cost:.4f}",
        )

    # ── Git Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ensure_git_repo(project_path: Path) -> None:
        """Initialize a git repo if one doesn't exist at project_path."""
        git_dir = project_path / ".git"
        if git_dir.is_dir():
            return
        subprocess.run(
            ["git", "init"],
            cwd=str(project_path),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=str(project_path),
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit (IMARO)", "--allow-empty"],
            cwd=str(project_path),
            capture_output=True,
            check=True,
        )
        logger.info("Initialized git repo at %s", project_path)

    @staticmethod
    def _commit_milestone(project_path: Path, milestone_id: str) -> None:
        """Stage all changes and commit after a milestone execution."""
        subprocess.run(
            ["git", "add", "."],
            cwd=str(project_path),
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(project_path),
            capture_output=True,
        )
        if result.returncode == 0:
            return  # Nothing staged
        subprocess.run(
            ["git", "commit", "-m", f"Milestone {milestone_id} completed (IMARO)"],
            cwd=str(project_path),
            capture_output=True,
            check=True,
        )
        logger.info("Committed milestone %s", milestone_id)

    # ── State Persistence ────────────────────────────────────────────────

    @staticmethod
    def _save_state(state_dir: Path, name: str, data: dict | list) -> None:
        path = state_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
