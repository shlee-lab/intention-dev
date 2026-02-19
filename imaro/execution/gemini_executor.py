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
- CRITICAL: Your entire response must be a single valid JSON object. \
Do not wrap it in markdown code fences. Do not add any text before or after the JSON.
"""

_JSON_REPAIR_PROMPT = """\
Your previous response was not valid JSON. I need you to reformat it.

Take the implementation you just described and output it as a SINGLE valid JSON object \
with this exact structure — nothing else:

{{
  "files": [
    {{"path": "relative/path/to/file.py", "action": "create", "content": "..."}},
    {{"path": "relative/path/to/file.py", "action": "edit", "content": "..."}}
  ],
  "summary": "Brief description of what was implemented"
}}

CRITICAL RULES:
- Output ONLY the JSON object — no markdown fences, no text before/after
- All string values must be properly JSON-escaped (newlines as \\n, quotes as \\", etc.)
- "content" must contain COMPLETE file contents

Here is the beginning of your previous response for reference:
{previous_response_preview}
"""

_FILE_LIST_PROMPT = """\
You are implementing a milestone for a project. List ONLY the files you need to \
create or modify. Do NOT generate any code yet.

{context}

Respond with ONLY a JSON object (no markdown fences):
{{"files": [{{"path": "relative/path/to/file.py", "action": "create"}}, ...]}}
"""

_SINGLE_FILE_PROMPT = """\
Generate the COMPLETE contents for this single file. Output ONLY the raw file \
contents — no JSON wrapping, no markdown fences, no explanation.

File: {file_path}
Action: {action}

{context}

{existing_content}

Output the complete file contents now — nothing else:\
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
                max_tokens=65536,
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

        logger.debug("Gemini execution response (first 1000 chars): %.1000s", response.content)
        result = self._apply_response(response.content, project_path, session_id)

        # If JSON parsing failed, attempt a repair pass
        if not result.success and "Could not parse JSON" in (result.error or ""):
            logger.warning("JSON parse failed, attempting repair pass")
            result = await self._retry_json_repair(
                response.content, project_path, session_id
            )

        # If repair also failed, fall back to per-file generation
        if not result.success and "Could not parse JSON" in (result.error or ""):
            logger.warning("JSON repair also failed, falling back to per-file generation")
            result = await self._execute_per_file(
                milestone, context, project_path, session_id
            )

        return result

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
                max_tokens=65536,
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

        result = self._apply_response(response.content, project_path, session_id)

        # If JSON parsing failed, attempt a repair pass
        if not result.success and "Could not parse JSON" in (result.error or ""):
            logger.warning("JSON parse failed in fix_issues, attempting repair pass")
            result = await self._retry_json_repair(
                response.content, project_path, session_id
            )

        # No per-file fallback for fix_issues — the repair is the last attempt

        return result

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

    async def _retry_json_repair(
        self, broken_response: str, project_path: Path, session_id: str
    ) -> ExecutionResult:
        """Re-prompt the LLM to output valid JSON when initial parse fails."""
        # Show a preview of the broken response (first 3000 chars)
        preview = broken_response[:3000]
        if len(broken_response) > 3000:
            preview += f"\n... (truncated, total length: {len(broken_response)})"

        prompt = _JSON_REPAIR_PROMPT.format(previous_response_preview=preview)

        try:
            response = await self._provider.generate(
                prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=65536,
            )
        except Exception as exc:
            logger.error("JSON repair LLM call failed: %s", exc)
            return ExecutionResult(
                success=False,
                error=f"JSON repair failed: {exc}",
                session_id=session_id,
            )

        logger.debug("JSON repair response (first 1000 chars): %.1000s", response.content)
        return self._apply_response(response.content, project_path, session_id)

    async def _execute_per_file(
        self,
        milestone: Milestone,
        context: str,
        project_path: Path,
        session_id: str,
    ) -> ExecutionResult:
        """Fallback: generate files one at a time to avoid large JSON responses."""
        # Step 1: Ask LLM for the list of files to create/modify
        milestone_ctx = (
            f"## Milestone: {milestone.name}\n"
            f"Description: {milestone.description}\n"
            f"Scope:\n" + "\n".join(f"  - {s}" for s in milestone.scope) + "\n"
            f"Acceptance Criteria:\n"
            + "\n".join(f"  - {c}" for c in milestone.acceptance_criteria)
        )
        full_context = f"## Project Context\n{context}\n\n{milestone_ctx}"

        list_prompt = _FILE_LIST_PROMPT.format(context=full_context)

        try:
            resp = await self._provider.generate(
                list_prompt, temperature=0.2, max_tokens=4096
            )
            data = _extract_json(resp.content)
            file_list = data.get("files", [])
        except Exception as exc:
            logger.error("Per-file fallback: file list generation failed: %s", exc)
            return ExecutionResult(
                success=False,
                error=f"Per-file fallback failed at file listing: {exc}",
                session_id=session_id,
            )

        if not file_list:
            return ExecutionResult(
                success=False,
                error="Per-file fallback: LLM returned empty file list",
                session_id=session_id,
            )

        logger.info(
            "Per-file fallback: generating %d files individually", len(file_list)
        )

        # Step 2: Generate each file individually
        applied: list[str] = []
        for file_entry in file_list:
            rel_path = file_entry.get("path", "")
            action = file_entry.get("action", "create")
            if not rel_path:
                continue

            # Check if file already exists and include its content as context
            existing = ""
            target = project_path / rel_path
            if target.is_file() and action == "edit":
                try:
                    old_content = target.read_text(encoding="utf-8")
                    existing = (
                        f"## Current contents of {rel_path}:\n"
                        f"```\n{old_content}\n```\n"
                        "Update this file — output the COMPLETE new contents."
                    )
                except (UnicodeDecodeError, OSError):
                    pass

            single_prompt = _SINGLE_FILE_PROMPT.format(
                file_path=rel_path,
                action=action,
                context=full_context,
                existing_content=existing,
            )

            try:
                file_resp = await self._provider.generate(
                    single_prompt,
                    system=(
                        "You are a senior software engineer. Output ONLY the raw "
                        "file contents. No JSON wrapping, no markdown fences, no "
                        "explanation text."
                    ),
                    temperature=0.3,
                    max_tokens=32768,
                )
            except Exception as exc:
                logger.error("Per-file fallback: failed to generate %s: %s", rel_path, exc)
                return ExecutionResult(
                    success=False,
                    output="\n".join(applied),
                    error=f"Per-file fallback: failed to generate {rel_path}: {exc}",
                    session_id=session_id,
                )

            file_content = file_resp.content

            # Strip markdown fences if present (LLM sometimes adds them despite instructions)
            file_content = _strip_markdown_fences(file_content)

            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_text(file_content, encoding="utf-8")
                applied.append(f"{action}: {rel_path}")
                logger.debug("Per-file applied %s: %s", action, rel_path)
            except Exception as exc:
                logger.error("Per-file fallback: failed to write %s: %s", rel_path, exc)
                return ExecutionResult(
                    success=False,
                    output="\n".join(applied),
                    error=f"Failed to write {rel_path}: {exc}",
                    session_id=session_id,
                )

        output_lines = [
            f"Per-file fallback: generated {len(applied)} files", "",
            "Files changed:",
        ] + applied
        return ExecutionResult(
            success=True,
            output="\n".join(output_lines),
            session_id=session_id,
        )

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

    # Strip markdown code fences (handle ```json, ```JSON, ``` etc.)
    if "```" in text:
        # Try all pairs of ``` fences (outermost first)
        fence_positions = []
        pos = 0
        while True:
            idx = text.find("```", pos)
            if idx == -1:
                break
            fence_positions.append(idx)
            pos = idx + 3

        # Try outermost pair
        if len(fence_positions) >= 2:
            start = fence_positions[0]
            # Skip the opening fence line (```json etc.)
            newline = text.find("\n", start)
            if newline != -1:
                inner_start = newline + 1
                inner_end = fence_positions[-1]
                candidate = text[inner_start:inner_end].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Try balanced brace matching from first { to find the complete JSON object
    brace_start = text.find("{")
    if brace_start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                if in_string:
                    escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # balanced but not valid JSON

    # Fallback: simple first-{ to last-} (less precise but catches more cases)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in response (length={len(text)})")


def _strip_markdown_fences(text: str) -> str:
    """Remove surrounding markdown code fences from raw file content."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence line
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()
