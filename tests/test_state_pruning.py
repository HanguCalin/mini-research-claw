"""Unit tests for backend/utils/state_pruning.py

Tests:
- build_scoped_view returns only allowed fields for each AI node
- build_scoped_view returns a full copy for unknown (non-AI) nodes
- the original state is never mutated
- NODE_SCOPE_CONFIG covers all expected AI nodes
- missing state keys are silently skipped (no KeyError)
"""

import pytest

from backend.utils.state_pruning import NODE_SCOPE_CONFIG, build_scoped_view


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def full_state():
    """A minimal but complete AutoResearchState for testing."""
    return {
        "topic": "Compare Random Forest vs XGBoost on tabular data",
        "arxiv_papers_full_text": [{"arxiv_id": "2401.00001", "title": "Paper A"}],
        "retrieval_round": 1,
        "kg_entities": [{"id": "e1", "canonical_name": "Random Forest"}],
        "kg_edges": [{"source_id": "e1", "target_id": "e2", "relation": "outperforms"}],
        "hypothesis": "RF outperforms XGBoost on small tabular datasets",
        "incremental_delta": "Focuses on datasets < 10k samples",
        "experiment_spec": {
            "independent_var": "model",
            "dependent_var": "accuracy",
            "control_description": "same train/test split",
            "dataset_id": "iris",
            "evaluation_metrics": ["accuracy"],
            "expected_outcome": "RF >= XGBoost",
        },
        "python_code": "print('hello')",
        "execution_logs": "stdout: hello",
        "metrics_json": {"accuracy": 0.95},
        "claim_ledger": [{"claim_id": "c1", "claim_text": "RF wins"}],
        "latex_draft": r"\section{Introduction} Hello.",
        "bibtex_source": "@article{ref1, title={A}}",
        "pipeline_status": "running",
    }


# ─── build_scoped_view — AI nodes ────────────────────────────────────────────


def test_kg_extractor_sees_only_papers(full_state):
    view = build_scoped_view(full_state, "kg_extractor")
    assert set(view.keys()) == {"arxiv_papers_full_text"}
    assert view["arxiv_papers_full_text"] == full_state["arxiv_papers_full_text"]


def test_hypothesis_generator_sees_correct_fields(full_state):
    view = build_scoped_view(full_state, "hypothesis_generator")
    assert set(view.keys()) == {"kg_entities", "kg_edges", "topic"}


def test_experiment_designer_sees_correct_fields(full_state):
    view = build_scoped_view(full_state, "experiment_designer")
    assert set(view.keys()) == {"hypothesis", "incremental_delta", "kg_entities", "kg_edges"}


def test_ml_coder_sees_only_spec_and_hypothesis(full_state):
    view = build_scoped_view(full_state, "ml_coder")
    assert set(view.keys()) == {"experiment_spec", "hypothesis"}
    # Must NOT see raw papers or KG
    assert "arxiv_papers_full_text" not in view
    assert "kg_entities" not in view
    assert "kg_edges" not in view


def test_ml_coder_retry_sees_debug_fields(full_state):
    view = build_scoped_view(full_state, "ml_coder_retry")
    assert set(view.keys()) == {"experiment_spec", "hypothesis", "python_code", "execution_logs"}


def test_academic_writer_excludes_raw_papers_and_logs(full_state):
    view = build_scoped_view(full_state, "academic_writer")
    assert set(view.keys()) == {
        "claim_ledger", "experiment_spec", "metrics_json", "incremental_delta", "hypothesis"
    }
    assert "arxiv_papers_full_text" not in view
    assert "execution_logs" not in view


def test_critique_panel_sees_correct_fields(full_state):
    view = build_scoped_view(full_state, "critique_panel")
    assert set(view.keys()) == {"latex_draft", "bibtex_source", "metrics_json", "claim_ledger"}


# ─── build_scoped_view — non-AI nodes ────────────────────────────────────────


def test_unknown_node_gets_full_state_copy(full_state):
    view = build_scoped_view(full_state, "executor")
    assert view == full_state


def test_unknown_node_returns_new_dict_not_same_object(full_state):
    view = build_scoped_view(full_state, "hitl_gate")
    assert view is not full_state


# ─── State immutability ───────────────────────────────────────────────────────


def test_original_state_not_mutated(full_state):
    original_topic = full_state["topic"]
    view = build_scoped_view(full_state, "hypothesis_generator")
    view["topic"] = "MODIFIED"
    assert full_state["topic"] == original_topic


# ─── Missing keys handled gracefully ─────────────────────────────────────────


def test_missing_state_keys_are_skipped():
    sparse_state = {"topic": "only topic present"}
    view = build_scoped_view(sparse_state, "hypothesis_generator")
    # Only "topic" is present in state; kg_entities and kg_edges are absent
    assert view == {"topic": "only topic present"}


def test_completely_empty_state():
    view = build_scoped_view({}, "ml_coder")
    assert view == {}


# ─── NODE_SCOPE_CONFIG completeness ──────────────────────────────────────────


def test_all_ai_nodes_are_in_scope_config():
    expected_ai_nodes = {
        "kg_extractor",
        "hypothesis_generator",
        "experiment_designer",
        "ml_coder",
        "ml_coder_retry",
        "academic_writer",
        "critique_panel",
    }
    assert expected_ai_nodes.issubset(set(NODE_SCOPE_CONFIG.keys()))


def test_each_node_scope_is_non_empty():
    for node_name, fields in NODE_SCOPE_CONFIG.items():
        assert len(fields) > 0, f"{node_name} has an empty scope — likely a config mistake"


def test_scope_fields_are_strings():
    for node_name, fields in NODE_SCOPE_CONFIG.items():
        for field in fields:
            assert isinstance(field, str), f"{node_name} scope contains non-string field: {field}"
