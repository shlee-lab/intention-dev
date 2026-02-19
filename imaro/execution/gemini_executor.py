"""Gemini-based milestone executor using LLM code generation."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from imaro.execution.base import Executor
from imaro.intention.document import IntentionDocumentManager
from imaro.models.schemas import ExecutionResult

if TYPE_CHECKING:
    from imaro.models.schemas import IntentionDocument, Milestone
    from imaro.providers.base import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior software engineer implementing code changes for a project.
You will be given a milestone description and project context.
You MUST respond with valid JSON only — no markdown fences, no extra text.

Respond with this exact JSON structure:
{
  "files": [
    {"path": "relative/path/to/file.py", "action": "create", "content": "file contents here"},
    {"path": "relative/path/to/file.py", "action": "edit", "content": "full new file contents"}
  ],
  "summary": "Brief description of what was implemented"
}

Rules:
- "action" must be "create" for new files or "edit" for modifying existing files
- "content" must contain the complete file contents (not a diff)
- Use relative paths from the project root
- Create all necessary directories implicitly
- Follow the acceptance criteria and constraints exactly
- Write production-quality code with proper error handling
"""


class GeminiExecutor(Executor):
    """Execute milestones by generating code via Gemini API."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        # In-memory conversation history keyed by session_id
        self._sessions: dict[str, list[dict[str, str]]] = {}

    async def execute_milestone(
        self,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
        session_id: str | None = None,
    ) -> ExecutionResult:
        session_id = session_id or str(uuid.uuid4())
        context = IntentionDocumentManager.to_milestone_context(intention, milestone)
        prompt = self._build_execution_prompt(milestone, context, project_path)

        # Store prompt in session history
        self._sessions[session_id] = [{"role": "user", "content": prompt}]

        try:
            response = await self._provider.generate(
                prompt,
                system=SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=16384,
            )
        except Exception as exc:
            logger.error("Gemini execution failed: %s", exc)
            return ExecutionResult(
                success=False,
                error=f"LLM call failed: {exc}",
                session_id=session_id,
            )

        # Store response in session history
        self._sessions[session_id].append(
            {"role": "assistant", "content": response.content}
        )

        return self._apply_response(response.content, project_path, session_id)

    async def fix_issues(
        self,
        fix_instructions: str,
        session_id: str,
        milestone: Milestone,
        intention: IntentionDocument,
        project_path: Path,
    ) -> ExecutionResult:
        context = IntentionDocumentManager.to_milestone_context(intention, milestone)

        # Build fix prompt with prior conversation context
        prior_context = ""
        if session_id in self._sessions:
            prior_turns = self._sessions[session_id]
            prior_context = "\n\n".join(
                f"[{turn['role']}]: {turn['content']}" for turn in prior_turns
            )

        prompt = (
            f"## Prior Context\n{prior_context}\n\n"
            f"## Project Intention\n{context}\n\n"
            f"## Fix Instructions\n{fix_instructions}\n\n"
            "Apply the fixes and respond with the same JSON format as before. "
            "Include ALL files that need changes (with complete contents)."
        )

        # Append to session history
        self._sessions.setdefault(session_id, []).append(
            {"role": "user", "content": fix_instructions}
        )

        try:
            response = await self._provider.generate(
                prompt,
                system=SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=16384,
            )
        except Exception as exc:
            logger.error("Gemini fix failed: %s", exc)
            return ExecutionResult(
                success=False,
                error=f"LLM call failed: {exc}",
                session_id=session_id,
            )

        self._sessions[session_id].append(
            {"role": "assistant", "content": response.content}
        )

        return self._apply_response(response.content, project_path, session_id)

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

    def _apply_response(
        self, content: str, project_path: Path, session_id: str
    ) -> ExecutionResult:
        """Parse the LLM JSON response and apply file operations."""
        try:
            data = _extract_json(content)
        except ValueError as exc:
            logger.error("Failed to parse Gemini response: %s", exc)
            return ExecutionResult(
                success=False,
                output=content,
                error=f"Could not parse JSON from response: {exc}",
                session_id=session_id,
            )

        files = data.get("files", [])
        summary = data.get("summary", "")
        applied: list[str] = []

        for file_op in files:
            rel_path = file_op.get("path", "")
            action = file_op.get("action", "create")
            file_content = file_op.get("content", "")

            if not rel_path:
                continue

            target = project_path / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)

            try:
                target.write_text(file_content, encoding="utf-8")
                applied.append(f"{action}: {rel_path}")
                logger.debug("Applied %s: %s", action, rel_path)
            except Exception as exc:
                logger.error("Failed to write %s: %s", rel_path, exc)
                return ExecutionResult(
                    success=False,
                    output="\n".join(applied),
                    error=f"Failed to write {rel_path}: {exc}",
                    session_id=session_id,
                )

        output_lines = [summary, "", "Files changed:"] + applied
        return ExecutionResult(
            success=True,
            output="\n".join(output_lines),
            session_id=session_id,
        )

    @staticmethod
    def _build_execution_prompt(
        milestone: Milestone, context: str, project_path: Path
    ) -> str:
        lines = [
            f"## Project Context\n{context}",
            "",
            f"## Milestone: {milestone.name}",
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
            lines.extend(
                ["", "Constraints:", *[f"  - {c}" for c in milestone.constraints]]
            )

        lines.extend([
            "",
            f"## Working Directory: {project_path}",
            "",
            "Implement this milestone now. Respond with JSON only.",
        ])
        return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may contain markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        # Find content between first ``` and last ```
        start = text.index("```")
        # Skip the opening fence line
        start = text.index("\n", start) + 1
        end = text.rindex("```")
        text = text[start:end].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object boundaries
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response (length={len(text)})")
