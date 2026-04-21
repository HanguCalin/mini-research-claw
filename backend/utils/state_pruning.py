"""Scoped state views — token-saving state pruning for AI-powered nodes.

AI nodes only see the fields they need, so prompts stay small and focused.
Non-AI nodes (Dependency Resolver, Executor, HITL Gates, Linter, Claim Ledger
Builder, Critique Aggregator) receive the full state because they don't incur
LLM token costs and may need any field for routing logic.

Sourced from IMPLEMENTATION_GUIDE.md §2.2.
"""

from __future__ import annotations

from typing import Any

from backend.state import AutoResearchState

NODE_SCOPE_CONFIG: dict[str, list[str]] = {
    "kg_extractor": [
        "arxiv_papers_full_text",
    ],
    "hypothesis_generator": [
        "kg_entities",
        "kg_edges",
        "topic",
    ],
    "experiment_designer": [
        "hypothesis",
        "incremental_delta",
        "kg_entities",
        "kg_edges",
    ],
    "ml_coder": [
        "experiment_spec",
        "hypothesis",
    ],
    "ml_coder_retry": [
        "experiment_spec",
        "hypothesis",
        "python_code",
        "execution_logs",
    ],
    "academic_writer": [
        "claim_ledger",
        "experiment_spec",
        "metrics_json",
        "incremental_delta",
        "hypothesis",
    ],
    "critique_panel": [
        "latex_draft",
        "bibtex_source",
        "metrics_json",
        "claim_ledger",
    ],
    # latex_repair is a special case — receives only extracted error context
    # (line number + snippet + error message), not state fields. The caller
    # constructs that payload directly rather than going through this config.
}


def build_scoped_view(
    state: AutoResearchState,
    node_name: str,
) -> dict[str, Any]:
    """Return a new dict containing only the fields ``node_name`` is allowed to see.

    The original *state* is **never mutated** — this is purely a
    prompt-construction concern.

    If *node_name* is not in ``NODE_SCOPE_CONFIG`` (i.e. it's a non-AI node),
    a shallow copy of the full state is returned so callers always get a
    fresh dict they can safely modify without touching the pipeline state.
    """
    allowed_fields = NODE_SCOPE_CONFIG.get(node_name)

    if allowed_fields is None:
        return dict(state)

    return {key: state[key] for key in allowed_fields if key in state}  # type: ignore[literal-required]
