"""IMARO CLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="imaro",
    help="Intention-based Multi-Agent Review Orchestrator",
    no_args_is_help=True,
)
console = Console()


@app.command()
def start(
    description: str = typer.Argument(
        ..., help="Project description (natural language)"
    ),
    project_path: Path = typer.Option(
        ".", "--project-path", "-p", help="Project directory"
    ),
    plan_agents: int = typer.Option(
        3, "--plan-agents", help="Number of planning agents"
    ),
    consensus_threshold: float = typer.Option(
        0.75, "--consensus-threshold", help="Minimum consensus score"
    ),
    max_refinement_rounds: int = typer.Option(
        3, "--max-refinement-rounds", help="Max refinement Q&A rounds"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging"
    ),
) -> None:
    """Start the IMARO pipeline for a project."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from imaro.config import IMAROConfig
    from imaro.orchestrator import Orchestrator

    config = IMAROConfig(
        plan_agents=plan_agents,
        consensus_threshold=consensus_threshold,
        max_refinement_rounds=max_refinement_rounds,
    )

    orchestrator = Orchestrator(config=config)
    result = asyncio.run(orchestrator.run(description, project_path))

    if result.success:
        console.print("\n[bold green]Pipeline completed successfully.[/]")
    else:
        console.print(f"\n[bold red]Pipeline failed:[/] {result.error}")
        raise typer.Exit(code=1)


@app.command()
def status(
    project_path: Path = typer.Option(
        ".", "--project-path", "-p", help="Project directory"
    ),
) -> None:
    """Show current IMARO pipeline state."""
    state_dir = project_path / ".imaro"
    if not state_dir.exists():
        console.print("[dim]No IMARO state found in this directory.[/]")
        raise typer.Exit()

    for f in sorted(state_dir.glob("*.json")):
        console.print(f"[bold]{f.stem}[/]: {f.stat().st_size} bytes")


@app.command()
def intention(
    project_path: Path = typer.Option(
        ".", "--project-path", "-p", help="Project directory"
    ),
) -> None:
    """Show the saved intention document."""
    from imaro.intention.document import IntentionDocumentManager
    from imaro.ui.terminal import TerminalUI

    mgr = IntentionDocumentManager()
    try:
        doc = mgr.load(project_path)
    except FileNotFoundError:
        console.print("[dim]No intention document found.[/]")
        raise typer.Exit()

    ui = TerminalUI()
    ui.display_intention(doc)


if __name__ == "__main__":
    app()
