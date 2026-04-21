"""Node 1: Iterative Full-Text ArXiv Retriever.

Type: Non-AI (pure Python).
Libraries: arxiv, supabase, sentence-transformers, scikit-learn (TF-IDF).

Round 1 uses the raw topic string. Rounds 2+ refine the query using TF-IDF
keywords from the hypothesis and high-edge-count KG entity names. Implements
cache-first retrieval via Supabase, arXiv ToS rate limiting, and convergence
(early exit if < 2 new papers).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import arxiv
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.config import THRESHOLDS
from backend.state import AutoResearchState
from backend.utils.embeddings import embed_single
from backend.utils.supabase_client import get_supabase

logger = logging.getLogger(__name__)


def arxiv_retriever(state: AutoResearchState) -> dict[str, Any]:
    """Fetch papers from arXiv (cache-first), return updated paper list + round."""
    topic = state.get("topic", "")
    retrieval_round = state.get("retrieval_round", 0)
    existing_papers: list[dict[str, Any]] = list(state.get("arxiv_papers_full_text", []))
    existing_ids = {p["arxiv_id"] for p in existing_papers}

    if retrieval_round == 0:
        query = topic
    else:
        query = _build_refined_query(state)

    max_results = THRESHOLDS.arxiv_results_per_round
    new_papers: list[dict[str, Any]] = []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    client = arxiv.Client()

    for result in client.results(search):
        aid = _extract_arxiv_id(result.entry_id)
        if aid in existing_ids:
            continue

        paper = _cache_first_fetch(result, aid)
        if paper:
            new_papers.append(paper)
            existing_ids.add(aid)

        time.sleep(THRESHOLDS.arxiv_rate_limit_seconds)

    existing_papers.extend(new_papers)
    logger.info("Round %d: added %d new papers (total %d)",
                retrieval_round, len(new_papers), len(existing_papers))

    return {
        "arxiv_papers_full_text": existing_papers,
        "retrieval_round": retrieval_round + 1,
    }


# ─── Internal helpers ────────────────────────────────────────────────────────


def _build_refined_query(state: AutoResearchState) -> str:
    """Build a hypothesis-driven query for rounds 2+."""
    parts: list[str] = []

    hypothesis = state.get("hypothesis", "")
    if hypothesis:
        keywords = _tfidf_keywords(hypothesis, top_n=5)
        parts.extend(keywords)

    kg_entities = state.get("kg_entities", [])
    if kg_entities:
        edge_count: dict[str, int] = {}
        for edge in state.get("kg_edges", []):
            edge_count[edge["source_id"]] = edge_count.get(edge["source_id"], 0) + 1
            edge_count[edge["target_id"]] = edge_count.get(edge["target_id"], 0) + 1

        ranked = sorted(kg_entities, key=lambda e: edge_count.get(e["id"], 0), reverse=True)
        for ent in ranked[:3]:
            parts.append(ent["canonical_name"])

    return " AND ".join(parts) if parts else state.get("topic", "")


def _tfidf_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top-N TF-IDF keywords from a single text."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    try:
        vectorizer.fit_transform([text])
        return list(vectorizer.get_feature_names_out())
    except ValueError:
        return text.split()[:top_n]


def _extract_arxiv_id(entry_id: str) -> str:
    """Extract the arXiv paper ID from the entry URL."""
    match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", entry_id)
    return match.group(1) if match else entry_id.split("/")[-1]


def _cache_first_fetch(
    result: arxiv.Result,
    arxiv_id: str,
) -> dict[str, Any] | None:
    """Check Supabase cache first; on miss, download and cache."""
    sb = get_supabase()

    cached = (
        sb.table("papers")
        .select("*")
        .eq("arxiv_id", arxiv_id)
        .limit(1)
        .execute()
    )
    if cached.data:
        row = cached.data[0]
        return {
            "arxiv_id": row["arxiv_id"],
            "title": row["title"],
            "authors": row["authors"],
            "year": row["year"],
            "abstract": row.get("abstract", ""),
            "full_text": row["full_text"],
            "embedding": row["embedding"],
        }

    full_text = _extract_sections(result)
    abstract = result.summary or ""
    embedding = embed_single(abstract)

    authors = [a.name for a in result.authors]
    year = result.published.year if result.published else 0

    try:
        sb.table("papers").insert({
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "full_text": full_text,
            "embedding": embedding,
        }).execute()
    except Exception:
        logger.warning("Failed to cache paper %s (duplicate?)", arxiv_id)

    return {
        "arxiv_id": arxiv_id,
        "title": result.title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "full_text": full_text,
        "embedding": embedding,
    }


def _extract_sections(result: arxiv.Result) -> dict[str, str]:
    """Extract methodology/implementation/results from the paper abstract.

    Full LaTeX source parsing would require downloading the .tar.gz; for now
    we use the abstract as a proxy and structure it into the expected schema.
    A future enhancement can download the full source for richer extraction.
    """
    summary = result.summary or ""
    return {
        "methodology": summary,
        "implementation": "",
        "results": "",
    }
