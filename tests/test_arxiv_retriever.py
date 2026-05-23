"""Unit tests for backend/agents/arxiv_retriever.py.

Tests:
- Round 1 uses raw topic.
- Rounds 2+ build refined queries using hypothesis and KG entities.
- Supabase cache hit returns cached paper.
- Supabase cache miss downloads and inserts paper.
- arXiv ID extraction handles various URL formats.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.arxiv_retriever import (
    _build_refined_query,
    _extract_arxiv_id,
    _tfidf_keywords,
    arxiv_retriever,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_supabase():
    mock = MagicMock()
    # Chainable mock for sb.table().select().eq().limit().execute()
    mock.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    return mock


@pytest.fixture
def sample_state():
    return {
        "topic": "quantum computing",
        "retrieval_round": 0,
        "arxiv_papers_full_text": [],
        "kg_entities": [],
        "kg_edges": [],
        "hypothesis": "",
    }


# ─── _extract_arxiv_id ────────────────────────────────────────────────────────

def test_extract_arxiv_id_standard():
    assert _extract_arxiv_id("http://arxiv.org/abs/2401.12345v1") == "2401.12345"
    assert _extract_arxiv_id("2401.12345v2") == "2401.12345"
    assert _extract_arxiv_id("2401.12345") == "2401.12345"


def test_extract_arxiv_id_old_format():
    assert _extract_arxiv_id("http://arxiv.org/abs/math/0501234") == "0501234"


# ─── _tfidf_keywords ──────────────────────────────────────────────────────────

def test_tfidf_keywords_extracts_words():
    text = "The quick brown fox jumps over the lazy dog"
    keywords = _tfidf_keywords(text, top_n=3)
    assert len(keywords) <= 3
    assert all(isinstance(k, str) for k in keywords)


# ─── _build_refined_query ─────────────────────────────────────────────────────

def test_build_refined_query_round_0_uses_topic():
    # Note: _build_refined_query is only called for rounds > 0 in the main loop,
    # but let's test its logic.
    state = {"topic": "AI", "hypothesis": "", "kg_entities": []}
    assert _build_refined_query(state) == "AI"


def test_build_refined_query_uses_hypothesis():
    state = {
        "topic": "AI",
        "hypothesis": "Deep learning for sentiment analysis in financial news",
        "kg_entities": []
    }
    query = _build_refined_query(state)
    assert "sentiment" in query.lower() or "financial" in query.lower()


def test_build_refined_query_uses_kg_entities():
    state = {
        "topic": "AI",
        "hypothesis": "",
        "kg_entities": [
            {"id": "e1", "canonical_name": "Transformer"},
            {"id": "e2", "canonical_name": "BERT"}
        ],
        "kg_edges": [
            {"source_id": "e1", "target_id": "e2"}
        ]
    }
    query = _build_refined_query(state)
    assert "Transformer" in query or "BERT" in query


# ─── arxiv_retriever Node ─────────────────────────────────────────────────────

@patch("backend.agents.arxiv_retriever.get_supabase")
@patch("backend.agents.arxiv_retriever.embed_single")
@patch("backend.agents.arxiv_retriever.arxiv.Client")
@patch("backend.agents.arxiv_retriever.arxiv.Search")
def test_arxiv_retriever_cache_hit(mock_search, mock_client, mock_embed, mock_get_sb, sample_state):
    # Mock arXiv result
    mock_result = MagicMock()
    mock_result.entry_id = "2401.00001"
    
    client_instance = mock_client.return_value
    client_instance.results.return_value = [mock_result]
    
    # Mock Supabase cache HIT
    sb = mock_get_sb.return_value
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {
            "arxiv_id": "2401.00001",
            "title": "Cached Paper",
            "authors": ["Author A"],
            "year": 2024,
            "abstract": "Abstract",
            "full_text": {"methodology": "Method"},
            "embedding": [0.1, 0.2]
        }
    ]
    
    result = arxiv_retriever(sample_state)
    
    assert len(result["arxiv_papers_full_text"]) == 1
    assert result["arxiv_papers_full_text"][0]["title"] == "Cached Paper"
    assert result["retrieval_round"] == 1
    # Should NOT have called insert
    assert not sb.table.return_value.insert.called


@patch("backend.agents.arxiv_retriever.get_supabase")
@patch("backend.agents.arxiv_retriever.embed_single")
@patch("backend.agents.arxiv_retriever.arxiv.Client")
@patch("backend.agents.arxiv_retriever.arxiv.Search")
def test_arxiv_retriever_cache_miss(mock_search, mock_client, mock_embed, mock_get_sb, sample_state):
    # Mock arXiv result
    mock_result = MagicMock()
    mock_result.entry_id = "2401.00002"
    mock_result.title = "New Paper"
    mock_result.authors = [MagicMock(name="Author B")]
    mock_result.published = MagicMock(year=2024)
    mock_result.summary = "New Summary"
    
    client_instance = mock_client.return_value
    client_instance.results.return_value = [mock_result]
    
    # Mock Supabase cache MISS
    sb = mock_get_sb.return_value
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    
    mock_embed.return_value = [0.3, 0.4]
    
    result = arxiv_retriever(sample_state)
    
    assert len(result["arxiv_papers_full_text"]) == 1
    assert result["arxiv_papers_full_text"][0]["title"] == "New Paper"
    # Should HAVE called insert
    assert sb.table.return_value.insert.called
