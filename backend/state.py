"""AutoResearchState and all supporting TypedDicts.

These are the data contracts every node reads from and writes to.
Each node function has the signature:

    def node_name(state: AutoResearchState) -> dict

and returns only the state keys it wants to update (LangGraph convention).

Sourced from IMPLEMENTATION_GUIDE.md §2.1.
"""

from __future__ import annotations

from typing import Any, TypedDict


# ─── Supporting TypedDicts (AutoResearchState depends on these) ──────────────


class KGEntity(TypedDict):
    """A single deduplicated knowledge graph node."""

    id: str
    canonical_name: str
    aliases: list[str]
    entity_type: str  # model | dataset | metric | method | hyperparameter
    attributes: dict[str, Any]


class KGEdge(TypedDict):
    """A directional, epistemic edge with polarity and boundary conditions."""

    source_id: str
    target_id: str
    relation: str  # outperforms | uses_dataset | achieves_metric | has_hyperparameter
    polarity: str  # supports | contradicts | neutral
    context_condition: str  # boundary condition; empty string if unconditional
    confidence: float
    provenance: str  # paper ID + section


class DebateEntry(TypedDict):
    """One round of the structured debate protocol (Phase 4)."""

    round: int
    challenger_role: str  # fact_checker | methodologist | formatter
    target_critique_index: int
    challenge: str
    response: str
    resolved: bool


class ClaimLedgerEntry(TypedDict):
    """Maps a single paper claim to its KG evidence (Node 5b)."""

    claim_id: str
    claim_text: str
    supporting_kg_edges: list[str]  # edge IDs
    contradicting_kg_edges: list[str]  # edge IDs
    evidence_strength: str  # strong | moderate | weak | unsupported


class ExperimentSpec(TypedDict):
    """The human-approved experiment contract (Node 3c → HITL Gate 2)."""

    independent_var: str
    dependent_var: str
    control_description: str
    dataset_id: str
    evaluation_metrics: list[str]
    expected_outcome: str


# ─── Main pipeline state ────────────────────────────────────────────────────


class AutoResearchState(TypedDict, total=False):
    """Full pipeline state threaded through every LangGraph node.

    `total=False` makes all keys optional — nodes only return the subset
    they want to update, and LangGraph merges the partial dict back.
    """

    # ── Phase 1: Literature + KG ─────────────────────────────────────────
    topic: str
    arxiv_papers_full_text: list[dict[str, Any]]
    retrieval_round: int
    kg_entities: list[KGEntity]
    kg_edges: list[KGEdge]
    hypothesis: str
    incremental_delta: str
    hypothesis_embedding: list[float]  # 384-dim SBERT vector
    novelty_score: float
    prior_art_similarity_score: float
    novelty_passed: bool

    # ── HITL Gates ───────────────────────────────────────────────────────
    hitl_approved: bool
    hitl_rejection_reason: str
    experiment_spec: ExperimentSpec
    hitl_experiment_approved: bool

    # ── Phase 2: Experimentation ─────────────────────────────────────────
    python_code: str
    resolved_dependencies: list[str]
    resolved_datasets: list[str]
    dataset_cache_path: str
    debug_instrumentation: str
    execution_success: bool
    execution_logs: str
    metrics_json: dict[str, Any]
    code_retry_count: int

    # ── Phases 3 & 4: Drafting + Critique ────────────────────────────────
    claim_ledger: list[ClaimLedgerEntry]
    latex_draft: str
    bibtex_source: str
    critique_warnings: list[dict[str, Any]]
    debate_log: list[DebateEntry]
    surviving_critiques: list[dict[str, Any]]
    confidence_score: float
    revision_pass_done: bool

    # ── Phase 5: Compilation ─────────────────────────────────────────────
    latex_compile_log: str
    latex_repair_attempts: int
    final_pdf_path: str | None

    # ── Supabase ─────────────────────────────────────────────────────────
    run_id: str
    artifact_urls: dict[str, str]

    # ── Telemetry ────────────────────────────────────────────────────────
    pipeline_status: str
    total_api_calls: int
    total_tokens_used: int
    logs: list[str]
