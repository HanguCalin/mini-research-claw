"""Singleton Supabase client.

Shared by:
  - Node 1 (ArXiv Retriever)         — paper cache lookups + inserts
  - Node 3 (Hypothesis Generator)    — pgvector prior-art cosine queries
  - Artifact Uploader                 — Storage uploads + pipeline_runs writes

Uses the **service-role** key (bypasses Row Level Security). Server-side only.
"""

from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from backend import config


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return the process-wide Supabase client.

    Cached so every node sees the same underlying HTTP session and connection
    pool. Raises a clear error if `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` are
    missing — failing here is preferable to failing inside a pipeline node.
    """
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Supabase credentials missing. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in your .env file."
        )
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)


def reset_client() -> None:
    """Drop the cached client (useful between tests)."""
    get_supabase.cache_clear()
