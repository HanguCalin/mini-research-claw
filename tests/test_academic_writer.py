"""Unit tests for backend/agents/academic_writer.py

Tests:
- first pass returns latex_draft and bibtex_source
- first pass only includes strong/moderate claims in prompt
- revision pass returns latex_draft and confidence_score
- revision pass sets revision_pass_done=True
- academic_writer routes to first vs revision pass based on revision_pass_done flag
- confidence_score defaults to 5.0 if missing from LLM response
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

from backend.agents.academic_writer import academic_writer

LATEX_RESPONSE = {
    "latex_draft": r"\documentclass{article}\begin{document}\section{Introduction}Hi.\end{document}",
    "bibtex_source": "@article{ref1, title={A}, author={B}, year={2024}}",
}

REVISION_RESPONSE = {
    "latex_draft": r"\documentclass{article}\begin{document}\section{Introduction}Revised.\end{document}",
    "confidence_score": 7.5,
}

BASE_STATE = {
    "hypothesis": "RF outperforms LR on tabular data.",
    "incremental_delta": "Focuses on small datasets.",
    "experiment_spec": {"dataset_id": "iris", "evaluation_metrics": ["accuracy"]},
    "metrics_json": {"accuracy": 0.95},
    "claim_ledger": [
        {"claim_id": "c1", "claim_text": "RF is better.", "evidence_strength": "strong"},
        {"claim_id": "c2", "claim_text": "LR is worse.", "evidence_strength": "moderate"},
        {"claim_id": "c3", "claim_text": "Results are universal.", "evidence_strength": "weak"},
    ],
    "revision_pass_done": False,
}


def _make_mock_client(response_json):
    block = MagicMock()
    block.text = json.dumps(response_json)
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ─── First pass ───────────────────────────────────────────────────────────────


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_first_pass_returns_latex_draft(mock_cls):
    mock_cls.return_value = _make_mock_client(LATEX_RESPONSE)
    result = academic_writer(BASE_STATE)
    assert "latex_draft" in result
    assert len(result["latex_draft"]) > 0


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_first_pass_returns_bibtex_source(mock_cls):
    mock_cls.return_value = _make_mock_client(LATEX_RESPONSE)
    result = academic_writer(BASE_STATE)
    assert "bibtex_source" in result


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_first_pass_excludes_weak_claims_from_prompt(mock_cls):
    mock_client = _make_mock_client(LATEX_RESPONSE)
    mock_cls.return_value = mock_client
    academic_writer(BASE_STATE)
    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Results are universal." not in user_content


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_first_pass_includes_strong_claims_in_prompt(mock_cls):
    mock_client = _make_mock_client(LATEX_RESPONSE)
    mock_cls.return_value = mock_client
    academic_writer(BASE_STATE)
    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "RF is better." in user_content


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_first_pass_does_not_set_revision_done(mock_cls):
    mock_cls.return_value = _make_mock_client(LATEX_RESPONSE)
    result = academic_writer(BASE_STATE)
    assert result["revision_pass_done"] is False


# ─── Revision pass ────────────────────────────────────────────────────────────


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_revision_pass_triggered_by_flag(mock_cls):
    mock_cls.return_value = _make_mock_client(REVISION_RESPONSE)
    state = {**BASE_STATE, "revision_pass_done": True,
             "latex_draft": LATEX_RESPONSE["latex_draft"],
             "critique_warnings": [], "surviving_critiques": []}
    result = academic_writer(state)
    assert result["revision_pass_done"] is True


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_revision_pass_returns_confidence_score(mock_cls):
    mock_cls.return_value = _make_mock_client(REVISION_RESPONSE)
    state = {**BASE_STATE, "revision_pass_done": True,
             "latex_draft": LATEX_RESPONSE["latex_draft"],
             "critique_warnings": [], "surviving_critiques": []}
    result = academic_writer(state)
    assert result["confidence_score"] == 7.5


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_revision_confidence_score_defaults_to_5(mock_cls):
    response_no_score = {"latex_draft": REVISION_RESPONSE["latex_draft"]}
    mock_cls.return_value = _make_mock_client(response_no_score)
    state = {**BASE_STATE, "revision_pass_done": True,
             "latex_draft": LATEX_RESPONSE["latex_draft"],
             "critique_warnings": [], "surviving_critiques": []}
    result = academic_writer(state)
    assert result["confidence_score"] == 5.0


@patch("backend.agents.academic_writer.anthropic.Anthropic")
def test_revision_includes_critiques_in_prompt(mock_cls):
    mock_client = _make_mock_client(REVISION_RESPONSE)
    mock_cls.return_value = mock_client
    state = {
        **BASE_STATE,
        "revision_pass_done": True,
        "latex_draft": LATEX_RESPONSE["latex_draft"],
        "critique_warnings": [{"message": "Missing methods section", "source": "linter"}],
        "surviving_critiques": [],
    }
    academic_writer(state)
    user_content = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Missing methods section" in user_content
