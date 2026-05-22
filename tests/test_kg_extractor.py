"""Unit tests for backend/agents/kg_extractor.py and backend/utils/llm_utils.py

kg_extractor tests:
- Entities and edges are extracted from a mocked Claude response
- Edges referencing unknown entity names are skipped
- Failed API calls per paper are skipped gracefully (other papers still processed)
- Node returns kg_entities and kg_edges keys
- _format_paper_for_prompt builds expected text block

llm_utils tests:
- extract_json handles clean JSON, markdown fences, and prose preambles
- extract_json raises ValueError on empty / non-JSON responses
- extract_text pulls text from Anthropic response object
- extract_text raises on empty content
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Block pyarrow / sentence_transformers before any backend import (Python 3.14 crash)
for _mod in [
    "pyarrow", "pyarrow.lib", "pyarrow.dataset",
    "sentence_transformers",
    "datasets",
    "backend.utils.embeddings",
]:
    sys.modules.setdefault(_mod, MagicMock())

from backend.agents.kg_extractor import _format_paper_for_prompt, kg_extractor
from backend.utils.llm_utils import extract_json, extract_text


# ─── Fixtures ─────────────────────────────────────────────────────────────────


VALID_KG_JSON = {
    "entities": [
        {"canonical_name": "Random Forest", "aliases": ["RF"],
         "entity_type": "model", "attributes": {}},
        {"canonical_name": "XGBoost", "aliases": ["XGB"],
         "entity_type": "model", "attributes": {}},
    ],
    "edges": [
        {
            "source_name": "Random Forest",
            "target_name": "XGBoost",
            "relation": "outperforms",
            "polarity": "supports",
            "context_condition": "",
            "confidence": 0.9,
        }
    ],
}

SAMPLE_PAPER = {
    "arxiv_id": "2401.00001",
    "title": "RF vs XGBoost",
    "abstract": "We compare two classifiers.",
    "full_text": {
        "methodology": "We use 5-fold CV.",
        "results": "RF wins on small datasets.",
    },
}


def _make_mock_client(response_json: dict | None = None, raise_exc: Exception | None = None):
    """Return a mock Anthropic client whose messages.create behaves as configured."""
    import json as _json

    client = MagicMock()
    if raise_exc is not None:
        client.messages.create.side_effect = raise_exc
    else:
        content_block = MagicMock()
        content_block.text = _json.dumps(response_json or VALID_KG_JSON)
        mock_response = MagicMock()
        mock_response.content = [content_block]
        client.messages.create.return_value = mock_response
    return client


# ─── _format_paper_for_prompt ─────────────────────────────────────────────────


def test_format_paper_includes_title():
    text = _format_paper_for_prompt(SAMPLE_PAPER)
    assert "RF vs XGBoost" in text


def test_format_paper_includes_arxiv_id():
    text = _format_paper_for_prompt(SAMPLE_PAPER)
    assert "2401.00001" in text


def test_format_paper_includes_abstract():
    text = _format_paper_for_prompt(SAMPLE_PAPER)
    assert "We compare two classifiers" in text


def test_format_paper_includes_full_text_sections():
    text = _format_paper_for_prompt(SAMPLE_PAPER)
    assert "5-fold CV" in text
    assert "RF wins" in text


def test_format_paper_handles_missing_fields():
    text = _format_paper_for_prompt({})
    assert "N/A" in text


# ─── kg_extractor node ────────────────────────────────────────────────────────


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_returns_required_keys(mock_anthropic_cls, mock_merge_kg):
    mock_anthropic_cls.return_value = _make_mock_client()
    mock_merge_kg.return_value = ([], [])

    state = {"arxiv_papers_full_text": [SAMPLE_PAPER]}
    result = kg_extractor(state)

    assert "kg_entities" in result
    assert "kg_edges" in result


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_creates_entities_from_response(mock_anthropic_cls, mock_merge_kg):
    mock_anthropic_cls.return_value = _make_mock_client()

    captured = {}
    def capture_merge(ex_ent, ex_edge, new_ent, new_edge, client):
        captured["new_entities"] = new_ent
        captured["new_edges"] = new_edge
        return new_ent, new_edge

    mock_merge_kg.side_effect = capture_merge

    kg_extractor({"arxiv_papers_full_text": [SAMPLE_PAPER]})

    names = [e["canonical_name"] for e in captured["new_entities"]]
    assert "Random Forest" in names
    assert "XGBoost" in names


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_creates_edge_with_correct_polarity(mock_anthropic_cls, mock_merge_kg):
    mock_anthropic_cls.return_value = _make_mock_client()

    captured = {}
    def capture_merge(ex_ent, ex_edge, new_ent, new_edge, client):
        captured["new_edges"] = new_edge
        return new_ent, new_edge

    mock_merge_kg.side_effect = capture_merge

    kg_extractor({"arxiv_papers_full_text": [SAMPLE_PAPER]})

    assert len(captured["new_edges"]) == 1
    assert captured["new_edges"][0]["polarity"] == "supports"
    assert captured["new_edges"][0]["relation"] == "outperforms"


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_skips_edge_with_unknown_entity(mock_anthropic_cls, mock_merge_kg):
    import anthropic as _anthropic

    bad_json = {
        "entities": [
            {"canonical_name": "Random Forest", "aliases": [],
             "entity_type": "model", "attributes": {}}
        ],
        "edges": [
            {
                "source_name": "Random Forest",
                "target_name": "UNKNOWN_ENTITY",
                "relation": "outperforms",
                "polarity": "supports",
                "context_condition": "",
                "confidence": 0.8,
            }
        ],
    }
    mock_anthropic_cls.return_value = _make_mock_client(response_json=bad_json)

    captured = {}
    def capture_merge(ex_ent, ex_edge, new_ent, new_edge, client):
        captured["new_edges"] = new_edge
        return new_ent, new_edge

    mock_merge_kg.side_effect = capture_merge

    kg_extractor({"arxiv_papers_full_text": [SAMPLE_PAPER]})

    assert captured["new_edges"] == []


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_skips_failed_paper_continues_others(mock_anthropic_cls, mock_merge_kg):
    import anthropic as _anthropic

    client = MagicMock()
    good_block = MagicMock()
    import json as _json
    good_block.text = _json.dumps(VALID_KG_JSON)
    good_response = MagicMock()
    good_response.content = [good_block]

    client.messages.create.side_effect = [
        _anthropic.APIError("boom", request=MagicMock(), body=None),
        good_response,
    ]
    mock_anthropic_cls.return_value = client

    captured = {}
    def capture_merge(ex_ent, ex_edge, new_ent, new_edge, cl):
        captured["new_entities"] = new_ent
        return new_ent, new_edge

    mock_merge_kg.side_effect = capture_merge

    paper2 = {**SAMPLE_PAPER, "arxiv_id": "2401.00002"}
    kg_extractor({"arxiv_papers_full_text": [SAMPLE_PAPER, paper2]})

    assert len(captured["new_entities"]) == 2


@patch("backend.agents.kg_extractor.merge_kg")
@patch("backend.agents.kg_extractor.anthropic.Anthropic")
def test_extractor_empty_papers_returns_empty_kg(mock_anthropic_cls, mock_merge_kg):
    mock_anthropic_cls.return_value = _make_mock_client()
    mock_merge_kg.return_value = ([], [])

    result = kg_extractor({"arxiv_papers_full_text": []})
    assert result["kg_entities"] == []
    assert result["kg_edges"] == []


# ─── extract_json ─────────────────────────────────────────────────────────────


def test_extract_json_clean():
    assert extract_json('{"key": "value"}') == {"key": "value"}


def test_extract_json_markdown_fenced():
    text = '```json\n{"key": "value"}\n```'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_prose_preamble():
    text = 'Here is the result:\n{"key": "value"}'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_trailing_prose():
    text = '{"key": "value"}\nThat is all.'
    assert extract_json(text) == {"key": "value"}


def test_extract_json_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        extract_json("")


def test_extract_json_raises_on_plain_text():
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def test_extract_json_raises_on_invalid_json():
    with pytest.raises(ValueError):
        extract_json("{bad json:}")


# ─── extract_text ─────────────────────────────────────────────────────────────


def test_extract_text_returns_text():
    block = MagicMock()
    block.text = "hello"
    response = MagicMock()
    response.content = [block]
    assert extract_text(response) == "hello"


def test_extract_text_raises_on_empty_content():
    response = MagicMock()
    response.content = []
    with pytest.raises(ValueError):
        extract_text(response)


def test_extract_text_raises_on_empty_text():
    block = MagicMock()
    block.text = ""
    response = MagicMock()
    response.content = [block]
    with pytest.raises(ValueError):
        extract_text(response)
