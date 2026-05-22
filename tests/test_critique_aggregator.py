"""Unit tests for backend/agents/critique_aggregator.py

Tests:
- Only linter warnings (source="linter") are kept from critique_warnings
- Debate-surviving critiques are merged in
- Duplicate messages are deduplicated
- revision_pass_done is set to True
- Empty inputs return empty merged list
"""

import pytest

from backend.agents.critique_aggregator import critique_aggregator


# ─── Helpers ──────────────────────────────────────────────────────────────────


def linter_warning(message):
    return {"source": "linter", "check": "imrad", "severity": "error", "message": message}


def agent_warning(message):
    return {"source": "critique_panel", "severity": "warning", "message": message}


def surviving_critique(critique_text):
    return {"role": "fact_checker", "critique": critique_text}


# ─── revision_pass_done ───────────────────────────────────────────────────────


def test_sets_revision_pass_done_true():
    result = critique_aggregator({})
    assert result["revision_pass_done"] is True


# ─── Linter warning filtering ─────────────────────────────────────────────────


def test_keeps_linter_warnings():
    state = {
        "critique_warnings": [linter_warning("Missing Introduction")],
        "surviving_critiques": [],
    }
    result = critique_aggregator(state)
    assert any(w.get("source") == "linter" for w in result["critique_warnings"])


def test_drops_non_linter_warnings():
    state = {
        "critique_warnings": [agent_warning("Some agent critique")],
        "surviving_critiques": [],
    }
    result = critique_aggregator(state)
    assert result["critique_warnings"] == []


def test_keeps_only_linter_from_mixed_warnings():
    state = {
        "critique_warnings": [
            linter_warning("Missing section"),
            agent_warning("Agent note"),
        ],
        "surviving_critiques": [],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1
    assert result["critique_warnings"][0]["source"] == "linter"


# ─── Surviving critiques merging ─────────────────────────────────────────────


def test_surviving_critiques_are_included():
    state = {
        "critique_warnings": [],
        "surviving_critiques": [surviving_critique("Citation X is not in the KG")],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1


def test_surviving_and_linter_both_included():
    state = {
        "critique_warnings": [linter_warning("Missing Methods")],
        "surviving_critiques": [surviving_critique("Claim Y is unsupported")],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 2


# ─── Deduplication ───────────────────────────────────────────────────────────


def test_duplicate_linter_messages_deduplicated():
    state = {
        "critique_warnings": [
            linter_warning("Missing Introduction"),
            linter_warning("Missing Introduction"),
        ],
        "surviving_critiques": [],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1


def test_duplicate_surviving_critiques_deduplicated():
    state = {
        "critique_warnings": [],
        "surviving_critiques": [
            surviving_critique("Same critique"),
            surviving_critique("Same critique"),
        ],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1


def test_linter_and_surviving_with_same_message_deduplicated():
    state = {
        "critique_warnings": [linter_warning("Duplicate message")],
        "surviving_critiques": [{"critique": "Duplicate message"}],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1


# ─── Empty inputs ─────────────────────────────────────────────────────────────


def test_empty_state_returns_empty_list():
    result = critique_aggregator({})
    assert result["critique_warnings"] == []


def test_empty_warnings_and_critiques_returns_empty():
    state = {"critique_warnings": [], "surviving_critiques": []}
    result = critique_aggregator(state)
    assert result["critique_warnings"] == []


# ─── surviving_critiques with "message" key instead of "critique" ─────────────


def test_surviving_critique_with_message_key():
    state = {
        "critique_warnings": [],
        "surviving_critiques": [{"message": "Alt key critique", "role": "methodologist"}],
    }
    result = critique_aggregator(state)
    assert len(result["critique_warnings"]) == 1
