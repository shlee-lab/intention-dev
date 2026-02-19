"""Tests for model validation and serialization."""

from __future__ import annotations

from imaro.models.schemas import (
    ConsensusResult,
    Contradiction,
    DriftReport,
    ExecutionResult,
    IntentionDocument,
    IntentionMapping,
    IssueSeverity,
    LLMResponse,
    Milestone,
    MilestoneResult,
    MilestoneStatus,
    Plan,
    ProjectResult,
    RefinementQuestion,
    RefinementRound,
    ReviewIssue,
    ReviewReport,
    ReviewResult,
    UserReviewDecision,
)


class TestModelInstantiation:
    """All key models can be instantiated with minimal or full data."""

    def test_refinement_question(self):
        q = RefinementQuestion(question="What framework?")
        assert q.question == "What framework?"
        assert q.context == ""

    def test_refinement_round(self):
        rnd = RefinementRound(
            questions=[RefinementQuestion(question="Q1")],
            answers=["A1"],
        )
        assert len(rnd.questions) == 1
        assert rnd.answers == ["A1"]

    def test_intention_document_defaults(self):
        doc = IntentionDocument()
        assert doc.purpose == ""
        assert doc.functional_requirements == []
        assert doc.refinement_history == []

    def test_intention_mapping_defaults(self):
        m = IntentionMapping()
        assert m.fulfills_requirements == []
        assert m.serves_purpose is False
        assert m.serves_core_value is False

    def test_milestone_defaults(self):
        m = Milestone(id="M1", name="Test", description="desc")
        assert m.estimated_complexity == "medium"
        assert m.depends_on == []
        assert isinstance(m.intention_mapping, IntentionMapping)

    def test_plan_defaults(self):
        p = Plan()
        assert p.milestones == []
        assert p.tech_stack == []

    def test_consensus_result_defaults(self):
        c = ConsensusResult()
        assert c.overall_score == 0.0
        assert c.recommendation == "proceed"

    def test_execution_result_defaults(self):
        e = ExecutionResult()
        assert e.success is False
        assert e.session_id == ""

    def test_review_issue(self):
        issue = ReviewIssue(description="Bug found")
        assert issue.severity == IssueSeverity.IMPROVEMENT
        assert issue.intention_relevant is False

    def test_review_result_defaults(self):
        r = ReviewResult()
        assert r.passed is True
        assert r.intention_alignment_score == 1.0

    def test_contradiction(self):
        c = Contradiction(
            reviewer_a="functionality",
            reviewer_b="edge_cases",
            description="Disagree on error handling",
        )
        assert c.resolution_hint == ""

    def test_review_report_defaults(self):
        rr = ReviewReport()
        assert rr.system_recommendation == ""

    def test_milestone_result_defaults(self):
        mr = MilestoneResult()
        assert mr.status == MilestoneStatus.PENDING
        assert mr.milestone is None

    def test_drift_report_defaults(self):
        d = DriftReport()
        assert d.still_aligned is True
        assert d.drift_score == 0.0

    def test_llm_response_defaults(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.usage == {}

    def test_project_result_defaults(self):
        pr = ProjectResult()
        assert pr.success is False
        assert pr.plans == []


class TestSerializationRoundTrip:
    """model_dump_json -> model_validate_json round-trip preserves data."""

    def test_intention_document(self, sample_intention):
        json_str = sample_intention.model_dump_json()
        restored = IntentionDocument.model_validate_json(json_str)
        assert restored == sample_intention

    def test_milestone(self, sample_milestone):
        json_str = sample_milestone.model_dump_json()
        restored = Milestone.model_validate_json(json_str)
        assert restored == sample_milestone

    def test_plan(self, sample_plan):
        json_str = sample_plan.model_dump_json()
        restored = Plan.model_validate_json(json_str)
        assert restored == sample_plan

    def test_consensus_result(self, sample_consensus):
        json_str = sample_consensus.model_dump_json()
        restored = ConsensusResult.model_validate_json(json_str)
        assert restored == sample_consensus

    def test_review_result_with_issues(self):
        r = ReviewResult(
            role="functionality",
            issues=[
                ReviewIssue(
                    description="Missing null check",
                    severity=IssueSeverity.CRITICAL,
                    file="main.py",
                    line=42,
                    suggestion="Add null check",
                    intention_relevant=True,
                )
            ],
            summary="Needs fixes",
            passed=False,
        )
        json_str = r.model_dump_json()
        restored = ReviewResult.model_validate_json(json_str)
        assert restored == r
        assert restored.issues[0].severity == IssueSeverity.CRITICAL


class TestEnumValues:
    """Enum values match expected strings."""

    def test_issue_severity_values(self):
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.IMPROVEMENT.value == "improvement"
        assert IssueSeverity.NITPICK.value == "nitpick"

    def test_user_review_decision_values(self):
        assert UserReviewDecision.APPROVE.value == "approve"
        assert UserReviewDecision.FIX.value == "fix"
        assert UserReviewDecision.SKIP.value == "skip"
        assert UserReviewDecision.ABORT.value == "abort"

    def test_milestone_status_values(self):
        assert MilestoneStatus.PENDING.value == "pending"
        assert MilestoneStatus.EXECUTING.value == "executing"
        assert MilestoneStatus.REVIEWING.value == "reviewing"
        assert MilestoneStatus.COMPLETED.value == "completed"
        assert MilestoneStatus.FAILED.value == "failed"
        assert MilestoneStatus.SKIPPED.value == "skipped"
