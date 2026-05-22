"""Unit tests for backend/agents/experiment_designer.py

Tests:
- node returns experiment_spec with all 6 required fields
- evaluation_metrics string is converted to list
- missing required fields raise ValueError
- prompt includes hypothesis, delta, KG entities and edges
"""

import json
import sys
from unittest.mock import MagicMock, patch

for _mod in [
    "pyarrow", "pyarrow.lib", "pyarrow.dataset",
    "sentence_transformers", "datasets", "backend.utils.embeddings",
]:
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from backend.agents.experiment_designer import REQUIRED_FIELDS, experiment_designer

VALID_SPEC_JSON = {
    "independent_var": "model_type",
    "dependent_var": "accuracy",
    "control_description": "same train/test split, random_state=42",
    "dataset_id": "sklearn.iris",
    "evaluation_metrics": ["accuracy", "f1_macro"],
    "expected_outcome": "RF >= LogisticRegression",
    "justifications": {
        "independent_var": "We vary model type.",
        "dependent_var": "Accuracy is the primary metric.",
        "control_description": "Fixed split ensures comparability.",
        "dataset_id": "Iris is a standard benchmark.",
        "evaluation_metrics": "Both accuracy and F1 matter.",
        "expected_outcome": "Based on KG evidence.",
    },
}

BASE_STATE = {
    "hypothesis": "RF outperforms LR on tabular data.",
    "incremental_delta": "Focuses on small datasets.",
    "kg_entities": [{"id": "e1", "canonical_name": "Random Forest"}],
    "kg_edges": [{
        "source_id": "e1", "target_id": "e2",
        "relation": "outperforms", "polarity": "supports",
        "context_condition": "", "confidence": 0.9, "provenance": "p1",
    }],
}


def _make_mock_client(response_json=None):
    block = MagicMock()
    block.text = json.dumps(response_json or VALID_SPEC_JSON)
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ─── experiment_designer node ─────────────────────────────────────────────────


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_returns_experiment_spec_key(mock_cls):
    mock_cls.return_value = _make_mock_client()
    result = experiment_designer(BASE_STATE)
    assert "experiment_spec" in result


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_spec_has_all_required_fields(mock_cls):
    mock_cls.return_value = _make_mock_client()
    result = experiment_designer(BASE_STATE)
    spec = result["experiment_spec"]
    for field in REQUIRED_FIELDS:
        assert field in spec, f"Missing field: {field}"


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_spec_fields_are_non_empty(mock_cls):
    mock_cls.return_value = _make_mock_client()
    result = experiment_designer(BASE_STATE)
    spec = result["experiment_spec"]
    for field in REQUIRED_FIELDS:
        assert spec[field], f"Field is empty: {field}"


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_evaluation_metrics_is_list(mock_cls):
    mock_cls.return_value = _make_mock_client()
    result = experiment_designer(BASE_STATE)
    assert isinstance(result["experiment_spec"]["evaluation_metrics"], list)


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_metrics_string_converted_to_list(mock_cls):
    spec_with_string_metrics = {**VALID_SPEC_JSON, "evaluation_metrics": "accuracy, f1"}
    mock_cls.return_value = _make_mock_client(spec_with_string_metrics)
    result = experiment_designer(BASE_STATE)
    assert isinstance(result["experiment_spec"]["evaluation_metrics"], list)
    assert "accuracy" in result["experiment_spec"]["evaluation_metrics"]


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_missing_required_field_raises_value_error(mock_cls):
    incomplete = {k: v for k, v in VALID_SPEC_JSON.items() if k != "dataset_id"}
    mock_cls.return_value = _make_mock_client(incomplete)
    with pytest.raises(ValueError, match="dataset_id"):
        experiment_designer(BASE_STATE)


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_prompt_includes_hypothesis(mock_cls):
    mock_client = _make_mock_client()
    mock_cls.return_value = mock_client
    experiment_designer(BASE_STATE)
    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "RF outperforms LR" in user_content


@patch("backend.agents.experiment_designer.anthropic.Anthropic")
def test_prompt_includes_kg_entity(mock_cls):
    mock_client = _make_mock_client()
    mock_cls.return_value = mock_client
    experiment_designer(BASE_STATE)
    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Random Forest" in user_content
