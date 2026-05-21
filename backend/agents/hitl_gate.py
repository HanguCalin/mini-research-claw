"""Node 3b: HITL Gate 1 — Hypothesis Approval.

Type: Non-AI (deterministic checkpoint).
Uses rich to render a formatted review panel, then blocks on input()
for operator approve/reject.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.config import THRESHOLDS
from backend.state import AutoResearchState

console = Console()


def hitl_gate(state: AutoResearchState) -> dict[str, Any]:
    """Display hypothesis for human review and block for approval."""
    hypothesis = state.get("hypothesis", "")
    incremental_delta = state.get("incremental_delta", "")
    novelty_score = state.get("novelty_score", 0.0)
    prior_art = state.get("prior_art_similarity_score", 0.0)
    kg_entities = state.get("kg_entities", [])
    kg_edges = state.get("kg_edges", [])
    papers = state.get("arxiv_papers_full_text", [])

    console.print("\n")
    console.print(Panel(
        hypothesis,
        title="[bold cyan]Generated Hypothesis[/bold cyan]",
        border_style="cyan",
    ))

    console.print(Panel(
        incremental_delta,
        title="[bold yellow]Incremental Delta (what's new)[/bold yellow]",
        border_style="yellow",
    ))

    score_table = Table(title="Novelty Scores")
    score_table.add_column("Metric", style="bold")
    score_table.add_column("Value")
    score_table.add_column("Threshold")
    score_table.add_column("Status")

    novelty_ok = novelty_score >= THRESHOLDS.novelty_threshold
    prior_ok = prior_art < THRESHOLDS.prior_art_ceiling

    score_table.add_row(
        "Novelty (RND)",
        f"{novelty_score:.3f}",
        f">= {THRESHOLDS.novelty_threshold}",
        "[green]PASS[/green]" if novelty_ok else "[red]FAIL[/red]",
    )
    score_table.add_row(
        "Prior-Art Similarity",
        f"{prior_art:.3f}",
        f"< {THRESHOLDS.prior_art_ceiling}",
        "[green]PASS[/green]" if prior_ok else "[red]FAIL[/red]",
    )
    console.print(score_table)

    if kg_edges:
        edge_table = Table(title="Key KG Triples (polarity)")
        edge_table.add_column("Source")
        edge_table.add_column("Relation")
        edge_table.add_column("Target")
        edge_table.add_column("Polarity")

        entity_map = {e["id"]: e["canonical_name"] for e in kg_entities}
        for edge in kg_edges[:10]:
            polarity_style = {
                "supports": "green",
                "contradicts": "red",
                "neutral": "dim",
            }.get(edge["polarity"], "dim")

            edge_table.add_row(
                entity_map.get(edge["source_id"], "?"),
                edge["relation"],
                entity_map.get(edge["target_id"], "?"),
                Text(edge["polarity"], style=polarity_style),
            )
        console.print(edge_table)

    if papers:
        paper_table = Table(title=f"Retrieved Papers ({len(papers)})")
        paper_table.add_column("ArXiv ID")
        paper_table.add_column("Title", max_width=60)

        for p in papers[:8]:
            paper_table.add_row(p.get("arxiv_id", "?"), p.get("title", "?"))
        if len(papers) > 8:
            paper_table.add_row("...", f"({len(papers) - 8} more)")
        console.print(paper_table)

    console.print("\n[bold]Enter [green]approve[/green] or [red]reject <reason>[/red]:[/bold]")
    user_input = input("> ").strip()

    if user_input.lower() == "approve":
        return {
            "hitl_approved": True,
            "pipeline_status": "approved_hypothesis",
        }

    reason = user_input.replace("reject", "", 1).strip() or "No reason provided"
    return {
        "hitl_approved": False,
        "hitl_rejection_reason": reason,
        "pipeline_status": "failed_hitl_rejected",
    }
