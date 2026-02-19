"""Intention document persistence and formatting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.models.schemas import IntentionDocument, Milestone

logger = logging.getLogger(__name__)

IMARO_DIR = ".imaro"
INTENTION_FILE = "intention.json"


class IntentionDocumentManager:
    """Save, load, and format IntentionDocument instances."""

    def save(self, doc: IntentionDocument, project_path: Path) -> Path:
        """Persist the intention document to .imaro/intention.json."""
        imaro_dir = project_path / IMARO_DIR
        imaro_dir.mkdir(parents=True, exist_ok=True)
        out = imaro_dir / INTENTION_FILE
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved intention document to %s", out)
        return out

    def load(self, project_path: Path) -> IntentionDocument:
        """Load an IntentionDocument from .imaro/intention.json."""
        from imaro.models.schemas import IntentionDocument as ID

        path = project_path / IMARO_DIR / INTENTION_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        return ID.model_validate(data)

    @staticmethod
    def to_prompt_context(doc: IntentionDocument) -> str:
        """Format the full intention document for LLM prompt inclusion."""
        lines = [
            "=== PROJECT INTENTION ===",
            f"Purpose: {doc.purpose}",
            f"Target Users: {doc.target_users}",
            f"Core Value: {doc.core_value}",
            "",
            "Functional Requirements:",
            *[f"  - {r}" for r in doc.functional_requirements],
            "",
            "Non-Functional Requirements:",
            *[f"  - {r}" for r in doc.non_functional_requirements],
            "",
            "Out of Scope:",
            *[f"  - {r}" for r in doc.out_of_scope],
            "",
            "Constraints:",
            *[f"  - {c}" for c in doc.constraints],
            "",
            "Success Criteria:",
            *[f"  - {s}" for s in doc.success_criteria],
            "=========================",
        ]
        return "\n".join(lines)

    @staticmethod
    def to_milestone_context(doc: IntentionDocument, milestone: Milestone) -> str:
        """Format intention context specific to a milestone."""
        mapping = milestone.intention_mapping
        lines = [
            "=== PROJECT INTENTION (milestone context) ===",
            f"Purpose: {doc.purpose}",
            f"Core Value: {doc.core_value}",
            f"Target Users: {doc.target_users}",
            "",
            f"--- Current Milestone: {milestone.name} ---",
            f"Description: {milestone.description}",
            "",
            "Scope:",
            *[f"  - {s}" for s in milestone.scope],
            "",
            "Acceptance Criteria:",
            *[f"  - {c}" for c in milestone.acceptance_criteria],
            "",
            "Constraints:",
            *[f"  - {c}" for c in milestone.constraints],
            "",
            "--- Intention Mapping ---",
            f"Serves Purpose: {mapping.serves_purpose}",
            f"Serves Core Value: {mapping.serves_core_value}",
            "Fulfills Requirements:",
            *[f"  - {r}" for r in mapping.fulfills_requirements],
            "Success Criteria Addressed:",
            *[f"  - {s}" for s in mapping.success_criteria_addressed],
            "============================================",
        ]
        return "\n".join(lines)
