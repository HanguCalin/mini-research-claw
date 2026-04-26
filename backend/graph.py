"""LangGraph orchestration — wires all 14 nodes into the pipeline DAG.

Node functions live in `backend/agents/`. This module is responsible only for:
  - registering each node with the graph,
  - defining unconditional edges,
  - defining the 6 conditional routing functions,
  - exposing `run_pipeline(topic)` with artifact-upload-on-exit semantics.

Sourced from IMPLEMENTATION_GUIDE.md §4.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend import config
from backend.config import THRESHOLDS
from backend.state import AutoResearchState
from backend.utils.artifact_uploader import (
    create_run,
    finalize_run,
    upload_artifacts,
)

# Phase 1 + HITL nodes
from backend.agents.arxiv_retriever import arxiv_retriever
from backend.agents.kg_extractor import kg_extractor
from backend.agents.hypothesis_generator import hypothesis_generator
from backend.agents.hitl_gate import hitl_gate
from backend.agents.experiment_designer import experiment_designer
from backend.agents.hitl_experiment_gate import hitl_experiment_gate

# Phase 2 nodes
from backend.agents.ml_coder import ml_coder
from backend.agents.dependency_resolver import dependency_resolver
from backend.agents.executor import executor

# Phase 3 + 4 nodes
from backend.agents.claim_ledger_builder import claim_ledger_builder
from backend.agents.academic_writer import academic_writer
from backend.agents.deterministic_linter import deterministic_linter
from backend.agents.critique_panel import critique_panel
from backend.agents.critique_aggregator import critique_aggregator

# Phase 5 node
from backend.agents.latex_compiler import latex_compiler

logger = logging.getLogger(__name__)


# ─── Conditional routing functions ───────────────────────────────────────────


def route_hypothesis(state: AutoResearchState) -> str:
    """After Node 3: regenerate / fail / loop / proceed."""
    if not state.get("kg_valid", True):
        return "hypothesis_generator"
    if not state.get("novelty_passed", False):
        return END
    if state.get("retrieval_round", 0) < THRESHOLDS.max_retrieval_rounds:
        return "arxiv_retriever"
    return "hitl_gate"


def route_hitl_hypothesis(state: AutoResearchState) -> str:
    """After Node 3b: experiment design or terminate."""
    if state.get("hitl_approved", False):
        return "experiment_designer"
    return END


def route_hitl_experiment(state: AutoResearchState) -> str:
    """After Node 3d: ml_coder / redesign / abort."""
    if state.get("hitl_experiment_approved", False):
        return "ml_coder"
    if state.get("pipeline_status") == "failed_hitl_rejected":
        return END
    return "experiment_designer"


def route_executor(state: AutoResearchState) -> str:
    """After Node 5: claim ledger / retry / fail."""
    if state.get("execution_success", False):
        return "claim_ledger_builder"
    if state.get("code_retry_count", 0) < THRESHOLDS.max_code_retries:
        return "ml_coder"
    return END


def route_claim_ledger(state: AutoResearchState) -> str:
    """After Node 5b: drafting or No-Paper exit."""
    if state.get("pipeline_status") == "no_paper":
        return END
    return "academic_writer"


def route_academic_writer(state: AutoResearchState) -> str:
    """After Node 6: critique loop (first pass) or compilation (revision pass)."""
    if state.get("revision_pass_done", False):
        return "latex_compiler"
    return "deterministic_linter"


# ─── Graph construction ──────────────────────────────────────────────────────


def build_graph() -> Any:
    """Build and compile the full LangGraph DAG."""
    graph = StateGraph(AutoResearchState)

    # ── Register all 14 nodes ────────────────────────────────────────────
    graph.add_node("arxiv_retriever", arxiv_retriever)
    graph.add_node("kg_extractor", kg_extractor)
    graph.add_node("hypothesis_generator", hypothesis_generator)
    graph.add_node("hitl_gate", hitl_gate)
    graph.add_node("experiment_designer", experiment_designer)
    graph.add_node("hitl_experiment_gate", hitl_experiment_gate)
    graph.add_node("ml_coder", ml_coder)
    graph.add_node("dependency_resolver", dependency_resolver)
    graph.add_node("executor", executor)
    graph.add_node("claim_ledger_builder", claim_ledger_builder)
    graph.add_node("academic_writer", academic_writer)
    graph.add_node("deterministic_linter", deterministic_linter)
    graph.add_node("critique_panel", critique_panel)
    graph.add_node("critique_aggregator", critique_aggregator)
    graph.add_node("latex_compiler", latex_compiler)

    # ── Unconditional edges ──────────────────────────────────────────────
    graph.add_edge(START, "arxiv_retriever")
    graph.add_edge("arxiv_retriever", "kg_extractor")
    graph.add_edge("kg_extractor", "hypothesis_generator")
    graph.add_edge("experiment_designer", "hitl_experiment_gate")
    graph.add_edge("ml_coder", "dependency_resolver")
    graph.add_edge("dependency_resolver", "executor")
    graph.add_edge("deterministic_linter", "critique_panel")
    graph.add_edge("critique_panel", "critique_aggregator")
    graph.add_edge("critique_aggregator", "academic_writer")
    graph.add_edge("latex_compiler", END)

    # ── Conditional edges ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "hypothesis_generator", route_hypothesis,
        {
            "hypothesis_generator": "hypothesis_generator",
            "arxiv_retriever": "arxiv_retriever",
            "hitl_gate": "hitl_gate",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "hitl_gate", route_hitl_hypothesis,
        {"experiment_designer": "experiment_designer", END: END},
    )
    graph.add_conditional_edges(
        "hitl_experiment_gate", route_hitl_experiment,
        {
            "ml_coder": "ml_coder",
            "experiment_designer": "experiment_designer",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "executor", route_executor,
        {
            "claim_ledger_builder": "claim_ledger_builder",
            "ml_coder": "ml_coder",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "claim_ledger_builder", route_claim_ledger,
        {"academic_writer": "academic_writer", END: END},
    )
    graph.add_conditional_edges(
        "academic_writer", route_academic_writer,
        {
            "deterministic_linter": "deterministic_linter",
            "latex_compiler": "latex_compiler",
        },
    )

    return graph.compile()


# Compile once at module load. LangGraph compilation is cheap; this avoids
# re-walking the graph definition on every pipeline invocation.
_compiled_graph = None


def get_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─── Pipeline runner with artifact-upload finally hook ──────────────────────


def run_pipeline(topic: str) -> AutoResearchState:
    """End-to-end pipeline invocation.

    Allocates a `run_id`, executes the LangGraph DAG, and uploads artifacts to
    Supabase Storage on every terminal path (success OR failure OR crash) via
    a `finally` block — per IMPLEMENTATION_GUIDE §4.4 (the simpler approach).
    """
    config.assert_env_ready()

    run_id = create_run(topic)
    logger.info("Starting pipeline run_id=%s topic=%r", run_id, topic)

    initial_state: AutoResearchState = {
        "topic": topic,
        "run_id": run_id,
        "retrieval_round": 0,
        "code_retry_count": 0,
        "pipeline_status": "running",
        "logs": [],
        "total_api_calls": 0,
        "total_tokens_used": 0,
    }

    final_state: AutoResearchState = dict(initial_state)  # type: ignore[assignment]

    try:
        result = get_graph().invoke(initial_state)
        final_state = result  # type: ignore[assignment]
        if not final_state.get("pipeline_status") or final_state.get("pipeline_status") == "running":
            final_state["pipeline_status"] = "success"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline crashed: %s", exc)
        final_state["pipeline_status"] = f"crashed: {exc.__class__.__name__}: {exc}"
    finally:
        try:
            urls = upload_artifacts(final_state)
            final_state["artifact_urls"] = urls
        except Exception:  # noqa: BLE001
            logger.exception("Artifact upload failed")
        try:
            finalize_run(final_state)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to finalize pipeline_runs row")

    logger.info(
        "Pipeline finished run_id=%s status=%s",
        run_id, final_state.get("pipeline_status"),
    )
    return final_state
