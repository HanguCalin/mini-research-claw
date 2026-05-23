"""Unit tests for Supabase-related caching and search logic.

Focuses on:
- Paper cache hit/miss in ArXiv Retriever.
- pgvector similarity search in Hypothesis Generator.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.agents.arxiv_retriever import _cache_first_fetch
from backend.agents.hypothesis_generator import _pgvector_novelty_check


@pytest.fixture
def mock_arxiv_result():
    res = MagicMock()
    res.entry_id = "http://arxiv.org/abs/2401.12345v1"
    res.title = "Test Paper"
    res.authors = [MagicMock(name="Author 1")]
    res.published = MagicMock(year=2024)
    res.summary = "Test summary"
    return res


# ─── ArXiv Cache Tests ────────────────────────────────────────────────────────

@patch("backend.agents.arxiv_retriever.get_supabase")
@patch("backend.agents.arxiv_retriever.embed_single")
def test_cache_first_fetch_hit(mock_embed, mock_get_sb, mock_arxiv_result):
    sb = mock_get_sb.return_value
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {
            "arxiv_id": "2401.12345",
            "title": "Cached Title",
            "authors": ["Author 1"],
            "year": 2024,
            "abstract": "Cached abstract",
            "full_text": {"methodology": "Cached methodology"},
            "embedding": [0.1] * 384
        }
    ]
    
    result = _cache_first_fetch(mock_arxiv_result, "2401.12345")
    
    assert result["title"] == "Cached Title"
    assert result["arxiv_id"] == "2401.12345"
    # Should not call insert
    assert sb.table.return_value.insert.call_count == 0


@patch("backend.agents.arxiv_retriever.get_supabase")
@patch("backend.agents.arxiv_retriever.embed_single")
def test_cache_first_fetch_miss_and_insert(mock_embed, mock_get_sb, mock_arxiv_result):
    sb = mock_get_sb.return_value
    # Cache miss
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    
    mock_embed.return_value = [0.2] * 384
    
    result = _cache_first_fetch(mock_arxiv_result, "2401.12345")
    
    assert result["title"] == "Test Paper"
    assert result["embedding"] == [0.2] * 384
    # Should call insert
    assert sb.table.return_value.insert.called
    insert_data = sb.table.return_value.insert.call_args[0][0]
    assert insert_data["arxiv_id"] == "2401.12345"


# ─── pgvector Novelty Tests ───────────────────────────────────────────────────

@patch("backend.agents.hypothesis_generator.get_supabase")
def test_pgvector_novelty_check_success(mock_get_sb):
    sb = mock_get_sb.return_value
    sb.rpc.return_value.execute.return_value.data = [
        {"arxiv_id": "1", "similarity": 0.95},
        {"arxiv_id": "2", "similarity": 0.90},
    ]
    
    avg_dist, max_sim = _pgvector_novelty_check([0.1] * 384)
    
    assert max_sim == 0.95
    # avg_dist = ((1-0.95) + (1-0.90)) / 2 = (0.05 + 0.10) / 2 = 0.075
    assert pytest.approx(avg_dist) == 0.075


@patch("backend.agents.hypothesis_generator.get_supabase")
def test_pgvector_novelty_check_empty(mock_get_sb):
    sb = mock_get_sb.return_value
    sb.rpc.return_value.execute.return_value.data = []
    
    avg_dist, max_sim = _pgvector_novelty_check([0.1] * 384)
    
    assert avg_dist == 1.0
    assert max_sim == 0.0


@patch("backend.agents.hypothesis_generator.get_supabase")
def test_pgvector_novelty_check_excludes_current(mock_get_sb):
    sb = mock_get_sb.return_value
    # Return 3 results, but one should be excluded
    sb.rpc.return_value.execute.return_value.data = [
        {"arxiv_id": "exclude-me", "similarity": 0.99},
        {"arxiv_id": "keep-me", "similarity": 0.80},
    ]
    
    avg_dist, max_sim = _pgvector_novelty_check(
        [0.1] * 384, 
        exclude_arxiv_ids=["exclude-me"]
    )
    
    assert max_sim == 0.80
    assert avg_dist == 1.0 - 0.80
