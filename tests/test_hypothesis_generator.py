"""Unit tests for backend/agents/hypothesis_generator.py

Tests:
- _build_kg_summary renders entities and edges correctly
- node returns required state keys
- novelty_passed is True when scores are within thresholds
- novelty_passed is False and pipeline_status="failed_novelty" when below threshold
- ungrounded entities are detected and kg_valid=False set
- pg_vector returning no results defaults to (1.0, 0.0)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.hypothesis_generator import _build_kg_summary, hypothesis_generator

KG_ENTITIES = [
    {"id": "e1", "canonical_name": "Random Forest", "entity_type": "model",
     "aliases": ["RF"], "attributes": {}},
    {"id": "e2", "canonical_name": "XGBoost", "entity_type": "model",
     "aliases": ["XGB"], "attributes": {}},
]

KG_EDGES = [{
    "source_id": "e1", "target_id": "e2",
    "relation": "outperforms", "polarity": "supports",
    "context_condition": "", "confidence": 0.9, "provenance": "p1",
}]

BASE_STATE = {
    "topic": "Compare RF vs XGBoost on tabular data",
    "kg_entities": KG_ENTITIES,
    "kg_edges": KG_EDGES,
    "arxiv_papers_full_text": [],
}

VALID_LLM_RESPONSE = {
    "hypothesis": "Random Forest outperforms XGBoost on small tabular datasets.",
    "incremental_delta": "Focuses on datasets below 10k samples.",
    "mentioned_entities": ["Random Forest", "XGBoost"],
}


def _make_mock_client(response_json=None):
    block = MagicMock()
    block.text = json.dumps(response_json or VALID_LLM_RESPONSE)
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ─── _build_kg_summary ────────────────────────────────────────────────────────


def test_kg_summary_includes_entity_names():
    summary = _build_kg_summary(KG_ENTITIES, KG_EDGES)
    assert "Random Forest" in summary
    assert "XGBoost" in summary


def test_kg_summary_includes_relation():
    summary = _build_kg_summary(KG_ENTITIES, KG_EDGES)
    assert "outperforms" in summary


def test_kg_summary_includes_polarity():
    summary = _build_kg_summary(KG_ENTITIES, KG_EDGES)
    assert "supports" in summary


def test_kg_summary_includes_context_condition():
    edges_with_condition = [{**KG_EDGES[0], "context_condition": "only on small datasets"}]
    summary = _build_kg_summary(KG_ENTITIES, edges_with_condition)
    assert "only on small datasets" in summary


def test_kg_summary_empty_kg():
    summary = _build_kg_summary([], [])
    assert "Entities:" in summary
    assert "Edges:" in summary


# ─── hypothesis_generator node ────────────────────────────────────────────────


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_returns_required_keys(mock_cls, mock_embed, mock_novelty):
    mock_cls.return_value = _make_mock_client()
    result = hypothesis_generator(BASE_STATE)
    for key in ["hypothesis", "incremental_delta", "novelty_score",
                "prior_art_similarity_score", "novelty_passed"]:
        assert key in result


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_novelty_passed_when_scores_good(mock_cls, mock_embed, mock_novelty):
    mock_cls.return_value = _make_mock_client()
    result = hypothesis_generator(BASE_STATE)
    assert result["novelty_passed"] is True


@patch("backend.agents.hypothesis_generator.THRESHOLDS")
@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.0, 0.99))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_novelty_failed_sets_pipeline_status(mock_cls, mock_embed, mock_novelty, mock_thresholds):
    mock_thresholds.novelty_threshold = 0.5
    mock_thresholds.prior_art_ceiling = 0.8
    mock_cls.return_value = _make_mock_client()
    result = hypothesis_generator(BASE_STATE)
    assert result["novelty_passed"] is False
    assert result.get("pipeline_status") == "failed_novelty"


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_ungrounded_entity_sets_kg_valid_false(mock_cls, mock_embed, mock_novelty):
    response_with_ungrounded = {
        **VALID_LLM_RESPONSE,
        "mentioned_entities": ["Random Forest", "NONEXISTENT_ENTITY"],
    }
    mock_cls.return_value = _make_mock_client(response_with_ungrounded)
    result = hypothesis_generator(BASE_STATE)
    assert result.get("kg_valid") is False


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_alias_counts_as_grounded(mock_cls, mock_embed, mock_novelty):
    # "RF" is an alias of "Random Forest" — should not be ungrounded
    response = {**VALID_LLM_RESPONSE, "mentioned_entities": ["RF", "XGB"]}
    mock_cls.return_value = _make_mock_client(response)
    result = hypothesis_generator(BASE_STATE)
    assert result.get("kg_valid") is not False


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_hypothesis_embedding_has_correct_length(mock_cls, mock_embed, mock_novelty):
    mock_cls.return_value = _make_mock_client()
    result = hypothesis_generator(BASE_STATE)
    assert len(result["hypothesis_embedding"]) == 384


# ─── BUG-003 regression tests ─────────────────────────────────────────────────


def _make_mock_client_raw_text(raw_text: str):
    block = MagicMock()
    block.text = raw_text
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_malformed_json_does_not_crash(mock_cls, mock_embed, mock_novelty):
    """BUG-003: hypothesis_generator must not crash when LLM returns invalid JSON."""
    mock_cls.return_value = _make_mock_client_raw_text("This is not JSON at all.")
    result = hypothesis_generator(BASE_STATE)
    assert "hypothesis" in result
    assert len(result["hypothesis"]) > 0


@patch("backend.agents.hypothesis_generator._pgvector_novelty_check", return_value=(0.8, 0.2))
@patch("backend.agents.hypothesis_generator.embed_single", return_value=[0.1] * 384)
@patch("backend.agents.hypothesis_generator.anthropic.Anthropic")
def test_missing_hypothesis_key_does_not_crash(mock_cls, mock_embed, mock_novelty):
    """BUG-003: hypothesis_generator must not crash when JSON is valid but missing 'hypothesis' key."""
    mock_cls.return_value = _make_mock_client({"incremental_delta": "some delta"})
    result = hypothesis_generator(BASE_STATE)
    assert "hypothesis" in result
    assert len(result["hypothesis"]) > 0
