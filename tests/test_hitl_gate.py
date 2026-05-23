"""Unit tests for backend/agents/hitl_gate.py

Tests:
- "approve" input sets hitl_approved=True and correct pipeline_status
- "reject <reason>" input sets hitl_approved=False with reason captured
- bare "reject" sets a default reason
- pipeline_status is set correctly for both outcomes
"""

from unittest.mock import patch

import pytest

from backend.agents.hitl_gate import hitl_gate


# ─── Fixtures ─────────────────────────────────────────────────────────────────


BASE_STATE = {
    "hypothesis": "RF outperforms XGBoost on small tabular datasets.",
    "incremental_delta": "Focuses on datasets < 10k samples.",
    "novelty_score": 0.75,
    "prior_art_similarity_score": 0.3,
    "kg_entities": [],
    "kg_edges": [],
    "arxiv_papers_full_text": [],
}


# ─── Approve flow ─────────────────────────────────────────────────────────────


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_gate.console")
def test_approve_sets_hitl_approved_true(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["hitl_approved"] is True


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_gate.console")
def test_approve_sets_correct_pipeline_status(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["pipeline_status"] == "approved_hypothesis"


@patch("builtins.input", return_value="APPROVE")
@patch("backend.agents.hitl_gate.console")
def test_approve_is_case_insensitive(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["hitl_approved"] is True


# ─── Reject flow ──────────────────────────────────────────────────────────────


@patch("builtins.input", return_value="reject hypothesis is too broad")
@patch("backend.agents.hitl_gate.console")
def test_reject_sets_hitl_approved_false(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["hitl_approved"] is False


@patch("builtins.input", return_value="reject hypothesis is too broad")
@patch("backend.agents.hitl_gate.console")
def test_reject_captures_reason(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert "hypothesis is too broad" in result["hitl_rejection_reason"]


@patch("builtins.input", return_value="reject")
@patch("backend.agents.hitl_gate.console")
def test_bare_reject_uses_default_reason(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["hitl_rejection_reason"] == "No reason provided"


@patch("builtins.input", return_value="reject")
@patch("backend.agents.hitl_gate.console")
def test_reject_sets_failed_pipeline_status(mock_console, mock_input):
    result = hitl_gate(BASE_STATE)
    assert result["pipeline_status"] == "failed_hitl_rejected"


# ─── State fields used for display (no crash) ─────────────────────────────────


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_gate.console")
def test_gate_works_with_kg_edges(mock_console, mock_input):
    state = {
        **BASE_STATE,
        "kg_entities": [{"id": "e1", "canonical_name": "RF"}],
        "kg_edges": [{
            "source_id": "e1", "target_id": "e2",
            "relation": "outperforms", "polarity": "supports",
            "context_condition": "", "confidence": 0.9, "provenance": "p1",
        }],
    }
    result = hitl_gate(state)
    assert result["hitl_approved"] is True


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_gate.console")
def test_gate_works_with_papers(mock_console, mock_input):
    state = {
        **BASE_STATE,
        "arxiv_papers_full_text": [
            {"arxiv_id": "2401.00001", "title": "Paper A"},
            {"arxiv_id": "2401.00002", "title": "Paper B"},
        ],
    }
    result = hitl_gate(state)
    assert result["hitl_approved"] is True
