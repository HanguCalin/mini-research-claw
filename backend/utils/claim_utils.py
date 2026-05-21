"""Claim-ledger evidence rating logic for Node 5b (Claim Ledger Builder).

Rates each claim's evidence_strength based on supporting/contradicting KG edges
and context_condition awareness.
"""

from __future__ import annotations

from backend.config import THRESHOLDS
from backend.state import ClaimLedgerEntry, KGEdge


def rate_evidence_strength(
    supporting: list[KGEdge],
    contradicting: list[KGEdge],
    claim_is_unconditional: bool = True,
) -> str:
    """Compute evidence strength per the plan's rating rules.

    Rules:
      - "strong":      >= 2 supporting, 0 contradicting
      - "moderate":    1 supporting, or >= 2 supporting with >= 1 contradicting
      - "weak":        1 supporting with >= 1 contradicting
      - "unsupported": 0 supporting

    context_condition awareness: if a supporting edge has a non-empty
    context_condition but the claim is unconditional, effective support count
    is reduced (conditional evidence for an unconditional claim is weaker).
    """
    effective_supporting = 0
    for edge in supporting:
        if claim_is_unconditional and edge.get("context_condition", ""):
            effective_supporting += 0.5
        else:
            effective_supporting += 1.0

    n_contra = len(contradicting)

    if effective_supporting == 0:
        return "unsupported"
    if effective_supporting >= 2 and n_contra == 0:
        return "strong"
    if effective_supporting >= 2 and n_contra >= 1:
        return "moderate"
    if effective_supporting >= 1 and n_contra == 0:
        return "moderate"
    if effective_supporting >= 1 and n_contra >= 1:
        return "weak"

    return "unsupported"


def should_trigger_no_paper(claim_ledger: list[ClaimLedgerEntry]) -> bool:
    """Return True if > 50% of claims are weak or unsupported (No-Paper gate)."""
    if not claim_ledger:
        return True

    weak_or_unsupported = sum(
        1 for c in claim_ledger
        if c["evidence_strength"] in ("weak", "unsupported")
    )
    fraction = weak_or_unsupported / len(claim_ledger)
    return fraction > THRESHOLDS.no_paper_weak_fraction


def find_edges_for_claim(
    claim_text: str,
    kg_edges: list[KGEdge],
    kg_entity_names: dict[str, str],
) -> tuple[list[KGEdge], list[KGEdge]]:
    """Find KG edges that support or contradict a given claim.

    Simple heuristic: an edge is relevant if both its source and target entity
    canonical names appear in the claim text (case-insensitive).
    """
    claim_lower = claim_text.lower()
    supporting: list[KGEdge] = []
    contradicting: list[KGEdge] = []

    for edge in kg_edges:
        source_name = kg_entity_names.get(edge["source_id"], "").lower()
        target_name = kg_entity_names.get(edge["target_id"], "").lower()

        if not source_name or not target_name:
            continue
        if source_name not in claim_lower and target_name not in claim_lower:
            continue

        if edge["polarity"] == "supports":
            supporting.append(edge)
        elif edge["polarity"] == "contradicts":
            contradicting.append(edge)

    return supporting, contradicting
