"""Claude Code CLI executor."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from imaro.execution.base import Executor
from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import ExecutionResult, IntentionDocument, Milestone

logger = logging.getLogger(__name__)


class ClaudeCodeExecutor(Executor):
    """Execute milestones by calling the `claude` CLI tool."""

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        timeout: int = 600,
    ) -> None:
        self.allowed_tools = allowed_tools or [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        ]
        self.default_timeout = timeout

    async def execute_milestone(
        self,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
        session_id: str | None = None,
    ) -> ExecutionResult:
        system_context = IntentionDocumentManager.to_milestone_context(
            intention, milestone
        )
        prompt = self._build_execution_prompt(milestone)
        timeout = self._timeout_for(milestone.estimated_complexity)

        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--append-system-prompt", system_context,
            "--allowedTools", ",".join(self.allowed_tools),
            "--cwd", str(project_path),
        ]
        if session_id:
            cmd.extend(["--resume", session_id])

        return await self._run(cmd, timeout)

    async def fix_issues(
        self,
        fix_instructions: str,
        session_id: str,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
    ) -> ExecutionResult:
        system_context = IntentionDocumentManager.to_milestone_context(
            intention, milestone
        )
        timeout = self._timeout_for(milestone.estimated_complexity)

        cmd = [
            "claude",
            "-p", fix_instructions,
            "--output-format", "json",
            "--append-system-prompt", system_context,
            "--allowedTools", ",".join(self.allowed_tools),
            "--cwd", str(project_path),
            "--resume", session_id,
        ]

        return await self._run(cmd, timeout)

    async def get_diff(self, project_path: Path) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff",
            cwd=str(project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("git diff failed: %s", stderr.decode())
            return ""
        return stdout.decode()

    async def get_changed_artifacts(self, project_path: Path) -> dict[str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only",
            cwd=str(project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return {}

        artifacts: dict[str, str] = {}
        for line in stdout.decode().strip().splitlines():
            filepath = project_path / line.strip()
            if filepath.is_file():
                try:
                    artifacts[line.strip()] = filepath.read_text(encoding="utf-8")
                except Exception:
                    logger.warning("Could not read %s", filepath)
        return artifacts

    async def _run(self, cmd: list[str], timeout: int) -> ExecutionResult:
        logger.debug("Running: %s", " ".join(cmd[:4]) + " ...")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout}s",
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                error="'claude' CLI not found. Is Claude Code installed?",
            )

        raw = stdout.decode()
        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                output=raw,
                error=stderr.decode() or f"Exit code {proc.returncode}",
            )

        # Parse JSON output for session_id and result
        session_id = ""
        output_text = raw
        try:
            data = json.loads(raw)
            session_id = data.get("session_id", "")
            output_text = data.get("result", raw)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Could not parse JSON from claude output")

        return ExecutionResult(
            success=True,
            output=output_text,
            session_id=session_id,
        )

    @staticmethod
    def _build_execution_prompt(milestone: Milestone) -> str:
        lines = [
            f"Implement milestone: {milestone.name}",
            "",
            f"Description: {milestone.description}",
            "",
            "Scope:",
            *[f"  - {s}" for s in milestone.scope],
            "",
            "Acceptance Criteria:",
            *[f"  - {c}" for c in milestone.acceptance_criteria],
        ]
        if milestone.constraints:
            lines.extend(["", "Constraints:", *[f"  - {c}" for c in milestone.constraints]])
        return "\n".join(lines)

    def _timeout_for(self, complexity: str) -> int:
        return {
            "low": 300,
            "medium": 600,
            "high": 1200,
        }.get(complexity, self.default_timeout)
