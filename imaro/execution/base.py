"""Abstract base class for milestone executors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imaro.models.schemas import ExecutionResult, IntentionDocument, Milestone


class Executor(ABC):
    """Interface for executing milestones and retrieving artifacts."""

    @abstractmethod
    async def execute_milestone(
        self,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
        session_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a milestone and return the result."""
        ...

    @abstractmethod
    async def fix_issues(
        self,
        fix_instructions: str,
        session_id: str,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
    ) -> ExecutionResult:
        """Resume a session to fix issues found in review."""
        ...

    @abstractmethod
    async def get_diff(self, project_path: Path) -> str:
        """Return the git diff for the project."""
        ...

    @abstractmethod
    async def get_changed_artifacts(self, project_path: Path) -> dict[str, str]:
        """Return {filepath: content} for files changed since last commit."""
        ...
