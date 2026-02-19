"""Minimal terminal UI for IMARO Phase 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

if TYPE_CHECKING:
    from imaro.models.schemas import (
        ConsensusResult,
        DriftReport,
        IntentionDocument,
        Milestone,
        Plan,
        RefinementQuestion,
        ReviewReport,
        UserReviewDecision,
    )


class TerminalUI:
    """Rich-based terminal interface for user interactions."""

    def __init__(self) -> None:
        self.console = Console()

    # ── Progress ─────────────────────────────────────────────────────────

    def show_progress(self, phase: str, detail: str = "") -> None:
        msg = f"[bold cyan]▶ {phase}[/]"
        if detail:
            msg += f"  [dim]{detail}[/]"
        self.console.print(msg)

    def show_success(self, message: str) -> None:
        self.console.print(f"[bold green]✓[/] {message}")

    def show_error(self, message: str) -> None:
        self.console.print(f"[bold red]✗[/] {message}")

    def show_warning(self, message: str) -> None:
        self.console.print(f"[bold yellow]![/] {message}")

    # ── Intention ────────────────────────────────────────────────────────

    def display_intention(self, doc: IntentionDocument) -> None:
        lines = [
            f"**Purpose:** {doc.purpose}",
            f"**Target Users:** {doc.target_users}",
            f"**Core Value:** {doc.core_value}",
            "",
            "**Functional Requirements:**",
            *[f"- {r}" for r in doc.functional_requirements],
            "",
            "**Non-Functional Requirements:**",
            *[f"- {r}" for r in doc.non_functional_requirements],
            "",
            "**Out of Scope:**",
            *[f"- {r}" for r in doc.out_of_scope],
            "",
            "**Constraints:**",
            *[f"- {c}" for c in doc.constraints],
            "",
            "**Success Criteria:**",
            *[f"- {s}" for s in doc.success_criteria],
        ]
        self.console.print(Panel(
            Markdown("\n".join(lines)),
            title="[bold]Intention Document[/]",
            border_style="blue",
        ))

    def confirm_intention(self, doc: IntentionDocument) -> bool:
        self.display_intention(doc)
        return Confirm.ask("\n[bold]Approve this intention document?[/]")

    def edit_intention(self, doc: IntentionDocument) -> IntentionDocument:
        self.console.print(
            "\n[dim]Enter updated values (press Enter to keep current):[/]\n"
        )
        fields = [
            ("purpose", "Purpose"),
            ("target_users", "Target Users"),
            ("core_value", "Core Value"),
        ]
        updates: dict[str, str] = {}
        for field, label in fields:
            current = getattr(doc, field)
            val = Prompt.ask(f"  {label}", default=current)
            updates[field] = val

        list_fields = [
            ("functional_requirements", "Functional Requirements"),
            ("non_functional_requirements", "Non-Functional Requirements"),
            ("out_of_scope", "Out of Scope"),
            ("constraints", "Constraints"),
            ("success_criteria", "Success Criteria"),
        ]
        list_updates: dict[str, list[str]] = {}
        for field, label in list_fields:
            current = getattr(doc, field)
            self.console.print(f"\n  [bold]{label}[/] (current):")
            for i, item in enumerate(current, 1):
                self.console.print(f"    {i}. {item}")
            edit = Confirm.ask(f"  Edit {label}?", default=False)
            if edit:
                self.console.print(
                    "  Enter items one per line (empty line to finish):"
                )
                items: list[str] = []
                while True:
                    item = Prompt.ask("    >", default="")
                    if not item:
                        break
                    items.append(item)
                list_updates[field] = items

        return doc.model_copy(update={**updates, **list_updates})

    # ── Refinement ───────────────────────────────────────────────────────

    def ask_refinement_questions(
        self, questions: list[RefinementQuestion]
    ) -> list[str]:
        self.console.print("\n[bold]Clarifying Questions:[/]\n")
        answers: list[str] = []
        for i, q in enumerate(questions, 1):
            if q.context:
                self.console.print(f"  [dim]{q.context}[/]")
            answer = Prompt.ask(f"  {i}. {q.question}")
            answers.append(answer)
        return answers

    def ask_skip_refinement(self) -> bool:
        return Confirm.ask(
            "\n[dim]Skip remaining questions and proceed with current info?[/]",
            default=False,
        )

    # ── Consensus ────────────────────────────────────────────────────────

    def display_consensus(self, consensus: ConsensusResult) -> None:
        table = Table(title="Consensus Evaluation")
        table.add_column("Metric", style="bold")
        table.add_column("Score")
        table.add_row(
            "Plan Consensus", f"{consensus.plan_consensus_score:.2f}"
        )
        table.add_row(
            "Intention Alignment", f"{consensus.intention_alignment_score:.2f}"
        )
        table.add_row("Overall", f"[bold]{consensus.overall_score:.2f}[/]")
        table.add_row("Recommendation", consensus.recommendation)
        self.console.print(table)

        if consensus.aligned_points:
            self.console.print("\n[green]Aligned Points:[/]")
            for p in consensus.aligned_points:
                self.console.print(f"  • {p}")
        if consensus.divergent_points:
            self.console.print("\n[yellow]Divergent Points:[/]")
            for p in consensus.divergent_points:
                self.console.print(f"  • {p}")
        if consensus.intention_gaps:
            self.console.print("\n[red]Intention Gaps:[/]")
            for g in consensus.intention_gaps:
                self.console.print(f"  • {g}")

    def handle_consensus_failure(
        self,
        consensus: ConsensusResult,
        plans: list[Plan],
        intention: IntentionDocument,
    ) -> str:
        self.display_consensus(consensus)
        self.console.print(
            "\n[bold yellow]Consensus is below threshold.[/]\n"
        )
        options = {
            "1": "proceed_anyway",
            "2": "revise_intention",
            "3": "select_best_plan",
            "4": "regenerate_plans",
            "5": "abort",
        }
        self.console.print("  1. Proceed anyway")
        self.console.print("  2. Revise the intention")
        self.console.print("  3. Select the best plan manually")
        self.console.print("  4. Regenerate plans")
        self.console.print("  5. Abort")
        choice = Prompt.ask("  Choose", choices=list(options.keys()))
        return options[choice]

    # ── Milestones ───────────────────────────────────────────────────────

    def display_milestones(self, milestones: list[Milestone]) -> None:
        for ms in milestones:
            mapping = ms.intention_mapping
            reqs = ", ".join(mapping.fulfills_requirements) or "—"
            criteria = ", ".join(mapping.success_criteria_addressed) or "—"
            lines = [
                f"**Description:** {ms.description}",
                f"**Complexity:** {ms.estimated_complexity}",
                f"**Scope:** {', '.join(ms.scope) or '—'}",
                "",
                "**Acceptance Criteria:**",
                *[f"- {c}" for c in ms.acceptance_criteria],
                "",
                f"**Fulfills Requirements:** {reqs}",
                f"**Success Criteria Addressed:** {criteria}",
            ]
            if ms.depends_on:
                lines.append(f"**Depends On:** {', '.join(ms.depends_on)}")
            self.console.print(Panel(
                Markdown("\n".join(lines)),
                title=f"[bold]Milestone {ms.id}: {ms.name}[/]",
                border_style="cyan",
            ))

    def confirm_milestones(
        self, milestones: list[Milestone], intention: IntentionDocument
    ) -> bool:
        self.display_milestones(milestones)
        return Confirm.ask("\n[bold]Approve these milestones?[/]")

    # ── Review ───────────────────────────────────────────────────────────

    def present_review_report(
        self,
        milestone: Milestone,
        report: ReviewReport,
        intention: IntentionDocument,
    ) -> UserReviewDecision:
        from imaro.models.schemas import UserReviewDecision as URD

        self.console.print(Panel(
            f"[bold]{milestone.name}[/] — Review Report",
            border_style="magenta",
        ))

        if report.critical_issues:
            self.console.print(f"\n[bold red]Critical Issues ({len(report.critical_issues)}):[/]")
            for issue in report.critical_issues:
                loc = f" ({issue.file}:{issue.line})" if issue.file else ""
                self.console.print(f"  ✗ {issue.description}{loc}")
                if issue.suggestion:
                    self.console.print(f"    → {issue.suggestion}")

        if report.improvement_issues:
            self.console.print(
                f"\n[yellow]Improvements ({len(report.improvement_issues)}):[/]"
            )
            for issue in report.improvement_issues:
                self.console.print(f"  • {issue.description}")

        if report.nitpick_issues:
            self.console.print(
                f"\n[dim]Nitpicks ({len(report.nitpick_issues)}):[/]"
            )
            for issue in report.nitpick_issues:
                self.console.print(f"  · {issue.description}")

        if report.contradictions:
            self.console.print(
                f"\n[bold yellow]Contradictions ({len(report.contradictions)}):[/]"
            )
            for c in report.contradictions:
                self.console.print(
                    f"  ⚡ {c.reviewer_a} vs {c.reviewer_b}: {c.description}"
                )

        self.console.print(
            f"\n  System recommendation: [bold]{report.system_recommendation}[/]"
        )

        options = {"1": URD.APPROVE, "2": URD.FIX, "3": URD.SKIP, "4": URD.ABORT}
        self.console.print("\n  1. Approve and continue")
        self.console.print("  2. Fix issues and re-review")
        self.console.print("  3. Skip review (accept as-is)")
        self.console.print("  4. Abort")
        choice = Prompt.ask("  Choose", choices=list(options.keys()))
        return options[choice]

    # ── Drift ────────────────────────────────────────────────────────────

    def handle_drift(self, drift: DriftReport, intention: IntentionDocument) -> str:
        self.console.print(Panel(
            f"Drift Score: [bold]{drift.drift_score:.2f}[/]",
            title="[bold yellow]Drift Alert[/]",
            border_style="yellow",
        ))
        if drift.drifted_areas:
            self.console.print("[yellow]Drifted Areas:[/]")
            for area in drift.drifted_areas:
                self.console.print(f"  • {area}")
        if drift.recommendations:
            self.console.print("[dim]Recommendations:[/]")
            for rec in drift.recommendations:
                self.console.print(f"  → {rec}")

        options = {"1": "continue", "2": "adjust", "3": "abort"}
        self.console.print("\n  1. Continue as planned")
        self.console.print("  2. Adjust approach")
        self.console.print("  3. Abort")
        choice = Prompt.ask("  Choose", choices=list(options.keys()))
        return options[choice]

    # ── Plan selection ───────────────────────────────────────────────────

    def select_plan(self, plans: list[Plan]) -> int:
        self.console.print("\n[bold]Available Plans:[/]\n")
        for i, plan in enumerate(plans, 1):
            self.console.print(
                f"  {i}. [bold]{plan.name}[/] — {plan.description[:80]}"
            )
        choice = Prompt.ask(
            "  Select plan",
            choices=[str(i) for i in range(1, len(plans) + 1)],
        )
        return int(choice) - 1
