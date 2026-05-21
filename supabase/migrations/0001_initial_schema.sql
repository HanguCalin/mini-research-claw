-- ─── Auto-Mini-Claw — Initial Supabase schema ────────────────────────────────
-- Run this once in the Supabase SQL Editor (Dashboard → SQL Editor → New query)
-- or via the Supabase CLI: `supabase db push`.
--
-- Provisions:
--   1. The pgvector extension (cosine similarity over SBERT embeddings).
--   2. `papers`        — paper-corpus cache (Node 1 + Node 3 prior-art lookup).
--   3. `pipeline_runs` — run metadata + artifact pointers (Artifact Uploader).
--
-- The `artifacts` Storage bucket is NOT created here — buckets must be created
-- via the Dashboard (Storage → New bucket → "artifacts", private) or the JS/Py
-- Storage API. See docs/SETUP.md §3 for the click path.

BEGIN;

-- ── Step 1: Enable pgvector ─────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Step 2: Paper-corpus cache ──────────────────────────────────────────────
-- `arxiv_id` is UNIQUE so cache-hit logic is a single-key lookup.
-- `full_text` is JSONB so the section parser can store {methodology,
-- implementation, results} without an extra schema.
-- `embedding` is vector(384) — must match SBERT all-MiniLM-L6-v2 output dim.
CREATE TABLE IF NOT EXISTS papers (
    id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    arxiv_id    TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    authors     TEXT[] NOT NULL,
    year        INT NOT NULL,
    abstract    TEXT,
    full_text   JSONB NOT NULL,
    embedding   vector(384) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IVFFlat cosine index — sub-millisecond ANN search for the Node 3
-- prior-art screening query (`ORDER BY embedding <=> :hypothesis_embedding`).
-- `lists = 100` is appropriate for corpora up to ~100K rows; bump for larger.
CREATE INDEX IF NOT EXISTS papers_embedding_ivfflat_idx
    ON papers USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ── Step 3: Pipeline runs ───────────────────────────────────────────────────
-- One row per `run_pipeline()` invocation. `status` transitions:
--   'running' → 'success' | 'failed_latex' | 'no_paper'
--             | 'failed_novelty' | 'failed_hitl' | 'failed_execution'
-- `artifact_path` is the Storage prefix `artifacts/{run_id}/`.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    topic         TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running',
    artifact_path TEXT,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at  TIMESTAMPTZ,
    metadata      JSONB
);

CREATE INDEX IF NOT EXISTS pipeline_runs_status_idx ON pipeline_runs (status);
CREATE INDEX IF NOT EXISTS pipeline_runs_started_at_idx ON pipeline_runs (started_at DESC);

COMMIT;
