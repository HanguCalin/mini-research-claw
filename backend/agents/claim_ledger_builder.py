"""Node 5b: Claim Ledger Builder.

Type: Non-AI (deterministic Python).
Enumerates claims from the hypothesis + experiment results, maps each to
KG evidence, rates evidence_strength, and triggers the No-Paper gate if
> 50% of claims are weak or unsupported.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from backend.state import AutoResearchState, ClaimLedgerEntry
from backend.utils.claim_utils import (
    find_edges_for_claim,
    rate_evidence_strength,
    should_trigger_no_paper,
)

logger = logging.getLogger(__name__)


def claim_ledger_builder(state: AutoResearchState) -> dict[str, Any]:
    """Build the claim ledger and enforce the No-Paper gate."""
    hypothesis = state.get("hypothesis", "")
    metrics = state.get("metrics_json", {})
    kg_entities = state.get("kg_entities", [])
    kg_edges = state.get("kg_edges", [])

    entity_names = {e["id"]: e["canonical_name"] for e in kg_entities}

    claims = _enumerate_claims(hypothesis, metrics)

    ledger: list[ClaimLedgerEntry] = []
    for claim_text in claims:
        supporting, contradicting = find_edges_for_claim(
            claim_text, kg_edges, entity_names,
        )

        strength = rate_evidence_strength(
            supporting, contradicting,
            claim_is_unconditional=True,
        )

        ledger.append(ClaimLedgerEntry(
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            claim_text=claim_text,
            supporting_kg_edges=[e.get("source_id", "") + "→" + e.get("target_id", "")
                                 for e in supporting],
            contradicting_kg_edges=[e.get("source_id", "") + "→" + e.get("target_id", "")
                                    for e in contradicting],
            evidence_strength=strength,
        ))

    if should_trigger_no_paper(ledger):
        logger.warning("No-Paper gate triggered: >50%% claims weak/unsupported")
        return {
            "claim_ledger": ledger,
            "pipeline_status": "no_paper",
            "final_pdf_path": None,
        }

    return {"claim_ledger": ledger}


def _enumerate_claims(
    hypothesis: str,
    metrics: dict[str, Any],
) -> list[str]:
    """Extract paper-level claims from the hypothesis text.

    Only the hypothesis sentences are enumerated — measurements like
    ``dbi=0.59`` are evidence, not claims. Treating each metric value as a
    separate claim used to inflate the ledger with ~30 unsupported entries
    that the Academic Writer never intended to assert, sinking the No-Paper
    gate even on successful runs. The full ``metrics_json`` is still
    available to the Writer for Results-section construction.

    *metrics* is accepted but not used; kept in the signature for backwards
    compatibility with callers and tests.
    """
    del metrics  # intentionally unused — see docstring

    claims: list[str] = []
    for sentence in hypothesis.replace(". ", ".\n").splitlines():
        sentence = sentence.strip()
        if len(sentence) > 20:
            claims.append(sentence)

    if not claims:
        claims.append(hypothesis)

    return claims
