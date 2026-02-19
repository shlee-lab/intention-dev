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
- "content" must contain the COMPLETE file contents (not a diff)
- For files that already exist, include the COMPLETE updated content — do not omit \
existing code that should be preserved
- Use relative paths from the project root
- Create all necessary directories implicitly
- Follow the acceptance criteria and constraints exactly
- Write production-quality code with proper error handling
"""

# Extensions to include when scanning existing project files
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".toml", ".yaml", ".yml",
    ".cfg", ".ini", ".md", ".txt", ".html", ".css", ".sh", ".sql",
})

# Directories to skip when scanning
_SKIP_DIRS = frozenset({
    ".git", ".imaro", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", "egg-info",
})

# Max size (bytes) for individual files included as context
_MAX_FILE_SIZE = 32_768

# Below this file count we include everything (no selection pass needed)
_SMALL_PROJECT_THRESHOLD = 15

_FILE_SELECTION_PROMPT = """\
You are a senior software engineer. Given a project file tree and a milestone \
description, select ONLY the files whose contents are needed to correctly \
implement the milestone. Include files that will be modified AND files that \
provide essential context (imports, schemas, configs, etc.).

## Milestone: {milestone_name}

Description: {milestone_description}

Scope:
{milestone_scope}

## Project File Tree
{file_tree}

Respond with ONLY a JSON object (no markdown fences):
{{"files": ["path/to/file1.py", "path/to/file2.py"]}}

Be selective — only include files that are truly necessary.\
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
        prompt = await self._build_execution_prompt(milestone, context, project_path)

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
        # Try diff against previous commit first (works after orchestrator commits)
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "HEAD~1",
            cwd=str(project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0 and stdout.strip():
            return stdout.decode()

        # Fallback: diff against working tree (unstaged changes)
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
        # Try diff against previous commit first
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "HEAD~1",
            cwd=str(project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0 or not stdout.strip():
            # Fallback: unstaged changes
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

    async def _build_execution_prompt(
        self, milestone: Milestone, context: str, project_path: Path
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

        # Include existing source files so the LLM preserves prior work
        file_tree = self._get_file_tree(project_path)
        if file_tree:
            selected = await self._select_relevant_files(
                file_tree, milestone, project_path
            )
            if selected:
                lines.append("")
                lines.append("## Existing Project Files")
                lines.append(
                    "The following files already exist. When modifying them, "
                    "include the COMPLETE updated content."
                )
                for rel_path, content in selected.items():
                    lines.append(f"\n### {rel_path}")
                    lines.append(f"```\n{content}\n```")

        lines.extend([
            "",
            f"## Working Directory: {project_path}",
            "",
            "Implement this milestone now. Respond with JSON only.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _get_file_tree(project_path: Path) -> list[str]:
        """Return sorted list of relative paths for all source files."""
        tree: list[str] = []
        for path in sorted(project_path.rglob("*")):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in _SOURCE_EXTENSIONS:
                continue
            if path.stat().st_size > _MAX_FILE_SIZE:
                continue
            try:
                tree.append(str(path.relative_to(project_path)))
            except ValueError:
                continue
        return tree

    async def _select_relevant_files(
        self,
        file_tree: list[str],
        milestone: Milestone,
        project_path: Path,
    ) -> dict[str, str]:
        """2-pass file selection: ask LLM which files matter, then read them.

        For small projects (< threshold files), skips the LLM call and
        includes everything directly.
        """
        # Small project — include all files, no selection needed
        if len(file_tree) <= _SMALL_PROJECT_THRESHOLD:
            return self._read_files(project_path, file_tree)

        # Pass 1: ask LLM to pick relevant files
        scope_text = "\n".join(f"  - {s}" for s in milestone.scope)
        prompt = _FILE_SELECTION_PROMPT.format(
            milestone_name=milestone.name,
            milestone_description=milestone.description,
            milestone_scope=scope_text,
            file_tree="\n".join(f"  {f}" for f in file_tree),
        )

        try:
            resp = await self._provider.generate(
                prompt, temperature=0.1, max_tokens=2048
            )
            data = _extract_json(resp.content)
            selected_paths = data.get("files", [])
            # Validate: only accept paths that actually exist in the tree
            tree_set = set(file_tree)
            selected_paths = [p for p in selected_paths if p in tree_set]
        except Exception as exc:
            logger.warning(
                "File selection LLM call failed (%s), falling back to all files", exc
            )
            selected_paths = file_tree

        if not selected_paths:
            selected_paths = file_tree

        logger.info(
            "File selection: %d/%d files selected for milestone %s",
            len(selected_paths), len(file_tree), milestone.id,
        )
        return self._read_files(project_path, selected_paths)

    @staticmethod
    def _read_files(project_path: Path, rel_paths: list[str]) -> dict[str, str]:
        """Read file contents for the given relative paths."""
        files: dict[str, str] = {}
        for rel in rel_paths:
            path = project_path / rel
            if not path.is_file():
                continue
            try:
                files[rel] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
        return files


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
