"""Node 3d: HITL Gate 2 — Experiment Approval.

Type: Non-AI (deterministic checkpoint).
Displays the ExperimentSpec for human review and blocks for approve/reject.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.state import AutoResearchState

console = Console()


def hitl_experiment_gate(state: AutoResearchState) -> dict[str, Any]:
    """Display experiment spec for human review; block for approval."""
    hypothesis = state.get("hypothesis", "")
    incremental_delta = state.get("incremental_delta", "")
    spec = state.get("experiment_spec")
    kg_edges = state.get("kg_edges", [])
    kg_entities = state.get("kg_entities", [])

    console.print("\n")
    console.print(Panel(hypothesis, title="[bold cyan]Hypothesis[/bold cyan]"))
    console.print(Panel(incremental_delta, title="[bold yellow]Incremental Delta[/bold yellow]"))

    if spec:
        spec_table = Table(title="Experiment Specification")
        spec_table.add_column("Field", style="bold")
        spec_table.add_column("Value")

        spec_table.add_row("Independent Variable", spec.get("independent_var", ""))
        spec_table.add_row("Dependent Variable", spec.get("dependent_var", ""))
        spec_table.add_row("Control Description", spec.get("control_description", ""))
        spec_table.add_row("Dataset ID", spec.get("dataset_id", ""))
        spec_table.add_row("Evaluation Metrics", ", ".join(spec.get("evaluation_metrics", [])))
        spec_table.add_row("Expected Outcome", spec.get("expected_outcome", ""))
        console.print(spec_table)

    contested_edges = [
        e for e in kg_edges if e.get("polarity") in ("supports", "contradicts")
    ]
    if contested_edges:
        entity_map = {e["id"]: e["canonical_name"] for e in kg_entities}
        edge_table = Table(title="Relevant KG Edges")
        edge_table.add_column("Source")
        edge_table.add_column("Relation")
        edge_table.add_column("Target")
        edge_table.add_column("Polarity")

        for edge in contested_edges[:8]:
            edge_table.add_row(
                entity_map.get(edge["source_id"], "?"),
                edge["relation"],
                entity_map.get(edge["target_id"], "?"),
                edge["polarity"],
            )
        console.print(edge_table)

    console.print("\n[bold]Enter [green]approve[/green] or [red]reject[/red] (or [red]abort[/red]):[/bold]")
    user_input = input("> ").strip().lower()

    if user_input == "approve":
        return {
            "hitl_experiment_approved": True,
            "pipeline_status": "approved_experiment",
        }

    if user_input == "abort":
        return {
            "hitl_experiment_approved": False,
            "pipeline_status": "failed_hitl_rejected",
        }

    return {
        "hitl_experiment_approved": False,
        "pipeline_status": "redesign_experiment",
    }
