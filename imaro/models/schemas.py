"""All Pydantic models for IMARO."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Intention ────────────────────────────────────────────────────────────────


class RefinementQuestion(BaseModel):
    question: str
    context: str = ""


class RefinementRound(BaseModel):
    questions: list[RefinementQuestion]
    answers: list[str]


class IntentionDocument(BaseModel):
    purpose: str = ""
    target_users: str = ""
    core_value: str = ""
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    raw_input: str = ""
    refinement_history: list[RefinementRound] = Field(default_factory=list)


# ── Planning ─────────────────────────────────────────────────────────────────


class IntentionMapping(BaseModel):
    fulfills_requirements: list[str] = Field(default_factory=list)
    serves_purpose: bool = False
    serves_core_value: bool = False
    success_criteria_addressed: list[str] = Field(default_factory=list)


class Milestone(BaseModel):
    id: str
    name: str
    description: str
    scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    estimated_complexity: str = "medium"  # low / medium / high
    intention_mapping: IntentionMapping = Field(default_factory=IntentionMapping)


class Plan(BaseModel):
    name: str = ""
    description: str = ""
    milestones: list[Milestone] = Field(default_factory=list)
    intention_alignment: dict[str, Any] = Field(default_factory=dict)
    architecture_notes: str = ""
    tech_stack: list[str] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    plan_consensus_score: float = 0.0
    intention_alignment_score: float = 0.0
    overall_score: float = 0.0
    aligned_points: list[str] = Field(default_factory=list)
    divergent_points: list[str] = Field(default_factory=list)
    intention_gaps: list[str] = Field(default_factory=list)
    recommendation: str = "proceed"  # proceed / advise_against / strongly_advise_against
    user_options: list[str] = Field(default_factory=list)


# ── Execution ────────────────────────────────────────────────────────────────


class ExecutionResult(BaseModel):
    success: bool = False
    output: str = ""
    session_id: str = ""
    error: str = ""


# ── Review ───────────────────────────────────────────────────────────────────


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    IMPROVEMENT = "improvement"
    NITPICK = "nitpick"


class ReviewIssue(BaseModel):
    description: str
    severity: IssueSeverity = IssueSeverity.IMPROVEMENT
    file: str = ""
    line: int | None = None
    suggestion: str = ""
    intention_relevant: bool = False


class ReviewResult(BaseModel):
    role: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
    intention_alignment_score: float = 1.0
    intention_alignment_details: str = ""
    passed: bool = True


class Contradiction(BaseModel):
    reviewer_a: str
    reviewer_b: str
    description: str
    resolution_hint: str = ""


class ReviewReport(BaseModel):
    reviews: list[ReviewResult] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    critical_issues: list[ReviewIssue] = Field(default_factory=list)
    improvement_issues: list[ReviewIssue] = Field(default_factory=list)
    nitpick_issues: list[ReviewIssue] = Field(default_factory=list)
    system_recommendation: str = ""  # approve / request_changes / discuss


class UserReviewDecision(str, Enum):
    APPROVE = "approve"
    FIX = "fix"
    SKIP = "skip"
    ABORT = "abort"


class MilestoneStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MilestoneResult(BaseModel):
    status: MilestoneStatus = MilestoneStatus.PENDING
    milestone: Milestone | None = None
    execution: ExecutionResult | None = None
    reviews: list[ReviewResult] = Field(default_factory=list)
    review_report: ReviewReport | None = None


# ── Drift ────────────────────────────────────────────────────────────────────


class DriftReport(BaseModel):
    drift_score: float = 0.0
    drifted_areas: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    still_aligned: bool = True


# ── LLM ──────────────────────────────────────────────────────────────────────


class LLMResponse(BaseModel):
    content: str = ""
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)


# ── Project ──────────────────────────────────────────────────────────────────


class ProjectResult(BaseModel):
    intention: IntentionDocument | None = None
    plans: list[Plan] = Field(default_factory=list)
    consensus: ConsensusResult | None = None
    milestones: list[Milestone] = Field(default_factory=list)
    milestone_results: list[MilestoneResult] = Field(default_factory=list)
    success: bool = False
    error: str = ""
