"""Unit tests for backend/agents/critique_panel.py

Tests:
- node returns critique_warnings, debate_log, surviving_critiques keys
- critiques from all 3 agents are included in warnings
- each critique gets a source field set to its agent role
- failed agent API call is skipped gracefully (other agents still run)
- retracted critiques are excluded from surviving_critiques
- debate_log is populated when challenges occur
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from backend.agents.critique_panel import critique_panel

BASE_STATE = {
    "latex_draft": r"\section{Introduction}Hello.\section{Methods}Stuff.",
    "bibtex_source": "@article{ref1, title={X}}",
    "metrics_json": {"accuracy": 0.95},
    "claim_ledger": [{"claim_id": "c1", "claim_text": "RF wins.", "evidence_strength": "strong"}],
    "kg_entities": [],
    "kg_edges": [],
    "critique_warnings": [],
}

CRITIQUE_A = [{"critique": "Claim X not in KG.", "severity": "error", "evidence": "edge e1"}]
CRITIQUE_B = [{"critique": "No confidence interval.", "severity": "warning", "evidence": ""}]
CRITIQUE_C = [{"critique": "Verbose intro.", "severity": "warning", "evidence": ""}]

NO_CHALLENGES = []
RETRACT_RESPONSE = {"action": "retract", "response": "You are right, I retract."}
DEFEND_RESPONSE = {"action": "defend", "response": "The claim is supported."}


def _text_block(obj):
    block = MagicMock()
    block.text = json.dumps(obj)
    resp = MagicMock()
    resp.content = [block]
    return resp


# ─── Basic output structure ───────────────────────────────────────────────────


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_returns_required_keys(mock_cls):
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    assert "critique_warnings" in result
    assert "debate_log" in result
    assert "surviving_critiques" in result


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_all_agent_critiques_in_warnings(mock_cls):
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    # 3 total critiques (1 per agent)
    new_warnings = [w for w in result["critique_warnings"]
                    if w.get("source") in ("fact_checker", "methodologist", "formatter")]
    assert len(new_warnings) == 3


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_each_critique_has_source_field(mock_cls):
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    for w in result["critique_warnings"]:
        assert "source" in w


# ─── Error resilience ─────────────────────────────────────────────────────────


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_failed_agent_skipped_others_still_run(mock_cls):
    import anthropic as _anthropic
    client = MagicMock()
    client.messages.create.side_effect = [
        _anthropic.APIError("boom", request=MagicMock(), body=None),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    # fact_checker failed → 0 critiques from it; methodologist + formatter ran
    non_empty_sources = {w["source"] for w in result["critique_warnings"]
                         if w.get("source") in ("methodologist", "formatter")}
    assert len(non_empty_sources) > 0


# ─── Debate protocol ──────────────────────────────────────────────────────────


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_retracted_critique_not_in_surviving(mock_cls):
    challenge = [{"target_critique_index": 0, "challenge": "That claim IS in the KG."}]
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),        # fact_checker
        _text_block(CRITIQUE_B),        # methodologist
        _text_block(CRITIQUE_C),        # formatter
        _text_block(NO_CHALLENGES),     # fact_checker challenges (none)
        _text_block(challenge),         # methodologist challenges fact_checker
        _text_block(NO_CHALLENGES),     # formatter challenges (none)
        _text_block(RETRACT_RESPONSE),  # fact_checker retracts
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    # The retracted critique should not appear in surviving_critiques
    surviving_texts = [c.get("critique") for c in result["surviving_critiques"]]
    assert "Claim X not in KG." not in surviving_texts


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_defended_critique_stays_in_surviving(mock_cls):
    challenge = [{"target_critique_index": 0, "challenge": "That claim IS in the KG."}]
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(challenge),
        _text_block(NO_CHALLENGES),
        _text_block(DEFEND_RESPONSE),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    surviving_texts = [c.get("critique") for c in result["surviving_critiques"]]
    assert "Claim X not in KG." in surviving_texts


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_debate_log_populated_on_challenge(mock_cls):
    challenge = [{"target_critique_index": 0, "challenge": "That claim IS in the KG."}]
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(challenge),
        _text_block(NO_CHALLENGES),
        _text_block(DEFEND_RESPONSE),
    ]
    mock_cls.return_value = client
    result = critique_panel(BASE_STATE)
    assert len(result["debate_log"]) >= 1


@patch("backend.agents.critique_panel.anthropic.Anthropic")
def test_existing_warnings_preserved(mock_cls):
    client = MagicMock()
    client.messages.create.side_effect = [
        _text_block(CRITIQUE_A),
        _text_block(CRITIQUE_B),
        _text_block(CRITIQUE_C),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
        _text_block(NO_CHALLENGES),
    ]
    mock_cls.return_value = client
    state = {**BASE_STATE, "critique_warnings": [
        {"source": "linter", "message": "existing warning"}
    ]}
    result = critique_panel(state)
    assert any(w.get("source") == "linter" for w in result["critique_warnings"])
