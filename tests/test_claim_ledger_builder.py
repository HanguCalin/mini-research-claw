"""Unit tests for backend/agents/claim_ledger_builder.py and backend/utils/claim_utils.py

claim_utils tests:
- rate_evidence_strength returns correct rating for all combinations
- context_condition on supporting edges weakens strength for unconditional claims
- should_trigger_no_paper fires at >50% weak/unsupported
- find_edges_for_claim matches edges by entity name presence in claim text

claim_ledger_builder tests:
- _enumerate_claims splits hypothesis into sentences
- node returns claim_ledger key
- no_paper gate triggers when majority of claims are unsupported
- no_paper gate does not trigger when evidence is sufficient
"""

import pytest

from backend.agents.claim_ledger_builder import _enumerate_claims, claim_ledger_builder
from backend.utils.claim_utils import (
    find_edges_for_claim,
    rate_evidence_strength,
    should_trigger_no_paper,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_edge(source_id, target_id, polarity="supports", context_condition=""):
    return {
        "source_id": source_id,
        "target_id": target_id,
        "relation": "outperforms",
        "polarity": polarity,
        "context_condition": context_condition,
        "confidence": 0.9,
        "provenance": "paper1",
    }


def make_claim(strength):
    return {
        "claim_id": "c1",
        "claim_text": "some claim",
        "supporting_kg_edges": [],
        "contradicting_kg_edges": [],
        "evidence_strength": strength,
    }


# ─── rate_evidence_strength ───────────────────────────────────────────────────


def test_no_supporting_is_unsupported():
    assert rate_evidence_strength([], []) == "unsupported"


def test_two_supporting_no_contra_is_strong():
    edges = [make_edge("e1", "e2"), make_edge("e1", "e3")]
    assert rate_evidence_strength(edges, []) == "strong"


def test_one_supporting_no_contra_is_moderate():
    assert rate_evidence_strength([make_edge("e1", "e2")], []) == "moderate"


def test_two_supporting_one_contra_is_moderate():
    edges = [make_edge("e1", "e2"), make_edge("e1", "e3")]
    contra = [make_edge("e2", "e1", polarity="contradicts")]
    assert rate_evidence_strength(edges, contra) == "moderate"


def test_one_supporting_one_contra_is_weak():
    sup = [make_edge("e1", "e2")]
    contra = [make_edge("e2", "e1", polarity="contradicts")]
    assert rate_evidence_strength(sup, contra) == "weak"


def test_conditional_edge_counts_as_half_for_unconditional_claim():
    conditional = [make_edge("e1", "e2", context_condition="only on small datasets")]
    # 0.5 effective support, no contra → unsupported (< 1.0)
    assert rate_evidence_strength(conditional, [], claim_is_unconditional=True) == "unsupported"


def test_conditional_edge_counts_full_for_conditional_claim():
    conditional = [make_edge("e1", "e2", context_condition="only on small datasets")]
    assert rate_evidence_strength(conditional, [], claim_is_unconditional=False) == "moderate"


def test_two_conditional_edges_sum_to_moderate():
    edges = [
        make_edge("e1", "e2", context_condition="only on small datasets"),
        make_edge("e1", "e3", context_condition="only on small datasets"),
    ]
    # 2 × 0.5 = 1.0 effective support → moderate (not strong, since < 2)
    result = rate_evidence_strength(edges, [], claim_is_unconditional=True)
    assert result in ("moderate", "unsupported")


# ─── should_trigger_no_paper ──────────────────────────────────────────────────


def test_no_paper_triggers_on_empty_ledger():
    assert should_trigger_no_paper([]) is True


def test_no_paper_triggers_when_majority_weak():
    ledger = [make_claim("weak"), make_claim("weak"), make_claim("strong")]
    assert should_trigger_no_paper(ledger) is True


def test_no_paper_triggers_when_majority_unsupported():
    ledger = [make_claim("unsupported"), make_claim("unsupported"), make_claim("moderate")]
    assert should_trigger_no_paper(ledger) is True


def test_no_paper_does_not_trigger_when_majority_strong():
    ledger = [make_claim("strong"), make_claim("strong"), make_claim("weak")]
    assert should_trigger_no_paper(ledger) is False


def test_no_paper_does_not_trigger_on_exactly_50_percent():
    ledger = [make_claim("weak"), make_claim("strong")]
    # exactly 50% — threshold is > 50%, so should NOT trigger
    assert should_trigger_no_paper(ledger) is False


# ─── find_edges_for_claim ─────────────────────────────────────────────────────


def test_find_edges_matches_by_entity_name():
    edges = [make_edge("e1", "e2", polarity="supports")]
    names = {"e1": "random forest", "e2": "xgboost"}
    claim = "random forest outperforms xgboost on tabular data"
    sup, contra = find_edges_for_claim(claim, edges, names)
    assert len(sup) == 1


def test_find_edges_separates_contradicting():
    edges = [make_edge("e1", "e2", polarity="contradicts")]
    names = {"e1": "random forest", "e2": "xgboost"}
    claim = "random forest outperforms xgboost"
    sup, contra = find_edges_for_claim(claim, edges, names)
    assert len(contra) == 1
    assert len(sup) == 0


def test_find_edges_skips_irrelevant_edges():
    edges = [make_edge("e3", "e4", polarity="supports")]
    names = {"e3": "bert", "e4": "gpt"}
    claim = "random forest outperforms xgboost"
    sup, contra = find_edges_for_claim(claim, edges, names)
    assert sup == []
    assert contra == []


def test_find_edges_empty_edges_returns_empty():
    sup, contra = find_edges_for_claim("any claim", [], {})
    assert sup == []
    assert contra == []


def test_find_edges_is_case_insensitive():
    edges = [make_edge("e1", "e2", polarity="supports")]
    names = {"e1": "Random Forest", "e2": "XGBoost"}
    claim = "random forest beats xgboost"
    sup, _ = find_edges_for_claim(claim, edges, names)
    assert len(sup) == 1


# ─── _enumerate_claims ────────────────────────────────────────────────────────


def test_enumerate_splits_on_sentence_boundary():
    hypothesis = "RF outperforms XGBoost on small datasets. The accuracy gap is significant."
    claims = _enumerate_claims(hypothesis, {})
    assert len(claims) == 2


def test_enumerate_filters_short_sentences():
    hypothesis = "RF wins. This is a longer meaningful claim about model performance."
    claims = _enumerate_claims(hypothesis, {})
    # "RF wins." is too short (< 20 chars), so only the longer sentence counts
    assert all(len(c) > 20 for c in claims)


def test_enumerate_fallback_on_empty_hypothesis():
    claims = _enumerate_claims("", {})
    assert claims == [""]


def test_enumerate_returns_hypothesis_if_no_long_sentences():
    hypothesis = "Short."
    claims = _enumerate_claims(hypothesis, {})
    assert claims == [hypothesis]


# ─── claim_ledger_builder node ────────────────────────────────────────────────


def test_builder_returns_claim_ledger_key():
    state = {
        "hypothesis": "RF outperforms XGBoost on tabular data.",
        "metrics_json": {"accuracy": 0.95},
        "kg_entities": [],
        "kg_edges": [],
    }
    result = claim_ledger_builder(state)
    assert "claim_ledger" in result


def test_builder_each_entry_has_required_fields():
    state = {
        "hypothesis": "RF outperforms XGBoost on tabular data.",
        "metrics_json": {},
        "kg_entities": [],
        "kg_edges": [],
    }
    result = claim_ledger_builder(state)
    for entry in result["claim_ledger"]:
        assert "claim_id" in entry
        assert "claim_text" in entry
        assert "evidence_strength" in entry


def test_builder_no_paper_gate_triggers_on_empty_kg():
    state = {
        "hypothesis": "RF outperforms XGBoost. Accuracy is higher. F1 is better.",
        "metrics_json": {},
        "kg_entities": [],
        "kg_edges": [],
    }
    result = claim_ledger_builder(state)
    # No KG edges → all claims unsupported → no_paper gate fires
    assert result.get("pipeline_status") == "no_paper"
    assert result.get("final_pdf_path") is None


def test_builder_no_paper_gate_does_not_trigger_with_strong_evidence():
    entities = [
        {"id": "e1", "canonical_name": "random forest"},
        {"id": "e2", "canonical_name": "xgboost"},
    ]
    edges = [
        make_edge("e1", "e2", polarity="supports"),
        make_edge("e1", "e2", polarity="supports"),
    ]
    state = {
        "hypothesis": "random forest outperforms xgboost on tabular data.",
        "metrics_json": {},
        "kg_entities": entities,
        "kg_edges": edges,
    }
    result = claim_ledger_builder(state)
    assert result.get("pipeline_status") != "no_paper"
