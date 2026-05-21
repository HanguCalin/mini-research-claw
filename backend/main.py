"""CLI entry point for the Auto-Mini-Claw pipeline.

Usage:
    mini-claw "Your research topic here"
    python -m backend.main "Your research topic here"
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.graph import run_pipeline

console = Console()


def cli() -> int:
    """Entry point registered in pyproject.toml [project.scripts]."""
    parser = argparse.ArgumentParser(
        prog="mini-claw",
        description="Auto-Mini-Claw — autonomous research paper pipeline.",
    )
    parser.add_argument(
        "topic",
        help='Research topic (e.g. "graph neural networks for fraud detection")',
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console.print(Panel(
        f"[bold]Topic:[/bold] {args.topic}",
        title="[bold cyan]Auto-Mini-Claw — Pipeline Start[/bold cyan]",
        border_style="cyan",
    ))

    final_state = run_pipeline(args.topic)
    _render_summary(final_state)

    status = final_state.get("pipeline_status", "unknown")
    return 0 if status == "success" else 1


def _render_summary(state: dict) -> None:
    """Pretty-print the terminal pipeline state."""
    status = state.get("pipeline_status", "unknown")
    style = "green" if status == "success" else "red"

    console.print()
    console.print(Panel(
        f"[bold {style}]{status.upper()}[/bold {style}]",
        title="Pipeline Result",
        border_style=style,
    ))

    summary = Table(title="Run Summary", show_header=False)
    summary.add_column("Field", style="bold")
    summary.add_column("Value")

    summary.add_row("Run ID", str(state.get("run_id", "—")))
    summary.add_row("Topic", str(state.get("topic", "—")))
    summary.add_row("Final PDF", str(state.get("final_pdf_path") or "—"))
    summary.add_row("Retrieval Rounds", str(state.get("retrieval_round", 0)))
    summary.add_row("Code Retries", str(state.get("code_retry_count", 0)))
    summary.add_row("LaTeX Repair Attempts", str(state.get("latex_repair_attempts", 0)))
    summary.add_row("Confidence Score", str(state.get("confidence_score", "—")))
    summary.add_row("Total API Calls", str(state.get("total_api_calls", 0)))

    console.print(summary)

    artifact_urls = state.get("artifact_urls") or {}
    if artifact_urls:
        artifacts = Table(title="Uploaded Artifacts")
        artifacts.add_column("File")
        artifacts.add_column("Storage Path")
        for filename, path in artifact_urls.items():
            artifacts.add_row(filename, path)
        console.print(artifacts)


if __name__ == "__main__":
    sys.exit(cli())
