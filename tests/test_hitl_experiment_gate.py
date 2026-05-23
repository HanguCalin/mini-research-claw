"""Unit tests for backend/agents/hitl_experiment_gate.py

Tests:
- "approve" sets hitl_experiment_approved=True and correct pipeline_status
- "reject" sets hitl_experiment_approved=False with redesign status
- "abort" sets hitl_experiment_approved=False with failed_hitl_rejected status
- node works with and without an experiment_spec in state
"""

from unittest.mock import patch

import pytest

from backend.agents.hitl_experiment_gate import hitl_experiment_gate

BASE_STATE = {
    "hypothesis": "RF outperforms XGBoost on small tabular datasets.",
    "incremental_delta": "Focuses on datasets < 10k samples.",
    "experiment_spec": {
        "independent_var": "model",
        "dependent_var": "accuracy",
        "control_description": "same train/test split",
        "dataset_id": "iris",
        "evaluation_metrics": ["accuracy"],
        "expected_outcome": "RF >= XGBoost",
    },
    "kg_entities": [],
    "kg_edges": [],
}


# ─── Approve ──────────────────────────────────────────────────────────────────


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_experiment_gate.console")
def test_approve_sets_approved_true(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["hitl_experiment_approved"] is True


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_experiment_gate.console")
def test_approve_sets_correct_pipeline_status(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["pipeline_status"] == "approved_experiment"


# ─── Reject (redesign) ────────────────────────────────────────────────────────


@patch("builtins.input", return_value="reject")
@patch("backend.agents.hitl_experiment_gate.console")
def test_reject_sets_approved_false(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["hitl_experiment_approved"] is False


@patch("builtins.input", return_value="reject")
@patch("backend.agents.hitl_experiment_gate.console")
def test_reject_routes_to_redesign(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["pipeline_status"] == "redesign_experiment"


# ─── Abort ────────────────────────────────────────────────────────────────────


@patch("builtins.input", return_value="abort")
@patch("backend.agents.hitl_experiment_gate.console")
def test_abort_sets_approved_false(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["hitl_experiment_approved"] is False


@patch("builtins.input", return_value="abort")
@patch("backend.agents.hitl_experiment_gate.console")
def test_abort_sets_failed_hitl_status(mock_console, mock_input):
    result = hitl_experiment_gate(BASE_STATE)
    assert result["pipeline_status"] == "failed_hitl_rejected"


# ─── Works without spec ───────────────────────────────────────────────────────


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_experiment_gate.console")
def test_works_without_experiment_spec(mock_console, mock_input):
    state = {**BASE_STATE, "experiment_spec": None}
    result = hitl_experiment_gate(state)
    assert result["hitl_experiment_approved"] is True


@patch("builtins.input", return_value="approve")
@patch("backend.agents.hitl_experiment_gate.console")
def test_works_with_kg_edges(mock_console, mock_input):
    state = {
        **BASE_STATE,
        "kg_entities": [{"id": "e1", "canonical_name": "RF"}],
        "kg_edges": [{
            "source_id": "e1", "target_id": "e2",
            "relation": "outperforms", "polarity": "supports",
            "context_condition": "", "confidence": 0.9, "provenance": "p1",
        }],
    }
    result = hitl_experiment_gate(state)
    assert result["hitl_experiment_approved"] is True
