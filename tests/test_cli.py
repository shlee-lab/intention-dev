"""CLI smoke tests using typer.testing.CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from imaro.cli import app
from imaro.models.schemas import IntentionDocument

runner = CliRunner()


def test_start_help():
    """'imaro start --help' exits 0 and shows help text."""
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
    assert "Start the IMARO pipeline" in result.output


def test_status_no_state(tmp_path):
    """'imaro status' on empty dir → 'No IMARO state found'."""
    result = runner.invoke(app, ["status", "--project-path", str(tmp_path)])
    assert result.exit_code == 0
    assert "No IMARO state found" in result.output


def test_status_with_state(tmp_path):
    """'imaro status' with .imaro/*.json → shows file listing."""
    state_dir = tmp_path / ".imaro"
    state_dir.mkdir()
    (state_dir / "intention.json").write_text('{"purpose": "test"}')
    (state_dir / "plans.json").write_text("[]")

    result = runner.invoke(app, ["status", "--project-path", str(tmp_path)])
    assert result.exit_code == 0
    assert "intention" in result.output
    assert "plans" in result.output


def test_intention_no_file(tmp_path):
    """'imaro intention' with no file → 'No intention document found'."""
    result = runner.invoke(app, ["intention", "--project-path", str(tmp_path)])
    assert result.exit_code == 0
    assert "No intention document found" in result.output


def test_intention_with_file(tmp_path):
    """'imaro intention' with saved document → displays it."""
    state_dir = tmp_path / ".imaro"
    state_dir.mkdir()
    doc = IntentionDocument(
        purpose="Build a test app",
        target_users="Testers",
        core_value="Quality",
        functional_requirements=["Run tests"],
        non_functional_requirements=["Fast"],
        out_of_scope=["Deployment"],
        constraints=["Python"],
        success_criteria=["All pass"],
        raw_input="test",
    )
    (state_dir / "intention.json").write_text(
        json.dumps(doc.model_dump(), indent=2)
    )

    result = runner.invoke(app, ["intention", "--project-path", str(tmp_path)])
    assert result.exit_code == 0
    assert "Build a test app" in result.output
