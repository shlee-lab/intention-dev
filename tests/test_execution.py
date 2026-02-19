"""Tests for ClaudeCodeExecutor and ContextManager."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imaro.execution.claude_code import ClaudeCodeExecutor
from imaro.execution.context_manager import ContextManager
from imaro.models.schemas import ExecutionResult


# ── ClaudeCodeExecutor ──────────────────────────────────────────────────────


def _make_mock_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Create a mock process with configurable output."""
    proc = AsyncMock()
    proc.communicate.return_value = (stdout, stderr)
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


class TestClaudeCodeExecutor:
    @pytest.mark.asyncio
    async def test_execute_milestone_builds_correct_args(
        self, sample_milestone, sample_intention, tmp_project
    ):
        output_data = json.dumps({"session_id": "sess-123", "result": "Done"})
        proc = _make_mock_process(stdout=output_data.encode(), returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            executor = ClaudeCodeExecutor()
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project
            )

            # Verify command args
            call_args = mock_exec.call_args[0]
            assert call_args[0] == "claude"
            assert "-p" in call_args
            assert "--output-format" in call_args
            assert "json" in call_args
            assert "--allowedTools" in call_args
            assert "--cwd" in call_args
            assert str(tmp_project) in call_args

    @pytest.mark.asyncio
    async def test_successful_execution(
        self, sample_milestone, sample_intention, tmp_project
    ):
        output_data = json.dumps({"session_id": "sess-abc", "result": "Implemented"})
        proc = _make_mock_process(stdout=output_data.encode(), returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            executor = ClaudeCodeExecutor()
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project
            )

            assert result.success is True
            assert result.session_id == "sess-abc"
            assert result.output == "Implemented"

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(
        self, sample_milestone, sample_intention, tmp_project
    ):
        proc = _make_mock_process(
            stdout=b"partial output",
            stderr=b"error occurred",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            executor = ClaudeCodeExecutor()
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project
            )

            assert result.success is False
            assert "error occurred" in result.error

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(
        self, sample_milestone, sample_intention, tmp_project
    ):
        proc = AsyncMock()
        proc.communicate.side_effect = asyncio.TimeoutError()
        proc.kill = MagicMock()

        with patch("asyncio.create_subprocess_exec", return_value=proc), \
             patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            executor = ClaudeCodeExecutor(timeout=1)
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project
            )

            assert result.success is False
            assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_file_not_found_graceful_error(
        self, sample_milestone, sample_intention, tmp_project
    ):
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            executor = ClaudeCodeExecutor()
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project
            )

            assert result.success is False
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fix_issues_includes_resume_flag(
        self, sample_milestone, sample_intention, tmp_project
    ):
        output_data = json.dumps({"session_id": "sess-fix", "result": "Fixed"})
        proc = _make_mock_process(stdout=output_data.encode(), returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            executor = ClaudeCodeExecutor()
            result = await executor.fix_issues(
                "Fix the null check",
                "sess-original",
                sample_milestone,
                sample_intention,
                tmp_project,
            )

            call_args = mock_exec.call_args[0]
            assert "--resume" in call_args
            assert "sess-original" in call_args
            assert result.success is True

    def test_timeout_for_returns_correct_values(self):
        executor = ClaudeCodeExecutor()
        assert executor._timeout_for("low") == 300
        assert executor._timeout_for("medium") == 600
        assert executor._timeout_for("high") == 1200
        # Unknown defaults to self.default_timeout
        assert executor._timeout_for("unknown") == executor.default_timeout

    def test_build_execution_prompt_includes_scope_and_criteria(
        self, sample_milestone
    ):
        prompt = ClaudeCodeExecutor._build_execution_prompt(sample_milestone)
        assert sample_milestone.name in prompt
        assert sample_milestone.description in prompt
        assert "Scope:" in prompt
        for s in sample_milestone.scope:
            assert s in prompt
        assert "Acceptance Criteria:" in prompt
        for c in sample_milestone.acceptance_criteria:
            assert c in prompt

    @pytest.mark.asyncio
    async def test_execute_milestone_with_session_id(
        self, sample_milestone, sample_intention, tmp_project
    ):
        """When session_id is provided, --resume flag is included."""
        output_data = json.dumps({"session_id": "sess-resume", "result": "Resumed"})
        proc = _make_mock_process(stdout=output_data.encode(), returncode=0)

        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            executor = ClaudeCodeExecutor()
            result = await executor.execute_milestone(
                sample_milestone, sample_intention, tmp_project, session_id="prev-sess"
            )

            call_args = mock_exec.call_args[0]
            assert "--resume" in call_args
            assert "prev-sess" in call_args


# ── ContextManager ──────────────────────────────────────────────────────────


class TestContextManager:
    def test_write_context_creates_claude_md(
        self, sample_milestone, sample_intention, tmp_project
    ):
        cm = ContextManager()
        cm.write_context(sample_milestone, sample_intention, tmp_project)

        claude_md = tmp_project / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "IMARO" in content
        assert sample_intention.purpose in content
        assert sample_intention.core_value in content
        assert sample_milestone.name in content
        assert "Acceptance Criteria" in content
        assert "Serves Purpose: True" in content

    def test_backup_context_copies_existing(self, tmp_project):
        claude_md = tmp_project / "CLAUDE.md"
        claude_md.write_text("Original content", encoding="utf-8")

        cm = ContextManager()
        cm.backup_context(tmp_project)

        backup = tmp_project / ".imaro" / "CLAUDE.md.backup"
        assert backup.exists()
        assert backup.read_text() == "Original content"

    def test_backup_context_no_existing_file(self, tmp_project):
        """No CLAUDE.md to back up — should not fail."""
        cm = ContextManager()
        cm.backup_context(tmp_project)
        backup = tmp_project / ".imaro" / "CLAUDE.md.backup"
        assert not backup.exists()

    def test_restore_context_from_backup(self, tmp_project):
        # Create a backup
        backup = tmp_project / ".imaro" / "CLAUDE.md.backup"
        backup.write_text("Original content", encoding="utf-8")

        # Create a current CLAUDE.md (would be overwritten)
        claude_md = tmp_project / "CLAUDE.md"
        claude_md.write_text("IMARO generated", encoding="utf-8")

        cm = ContextManager()
        cm.restore_context(tmp_project)

        assert claude_md.read_text() == "Original content"
        assert not backup.exists()  # Backup is cleaned up

    def test_restore_context_deletes_if_no_backup(self, tmp_project):
        """If no backup exists, CLAUDE.md is removed entirely."""
        claude_md = tmp_project / "CLAUDE.md"
        claude_md.write_text("IMARO generated", encoding="utf-8")

        cm = ContextManager()
        cm.restore_context(tmp_project)

        assert not claude_md.exists()

    def test_restore_context_no_file_no_backup(self, tmp_project):
        """Neither backup nor CLAUDE.md — should not fail."""
        cm = ContextManager()
        cm.restore_context(tmp_project)  # Should not raise
