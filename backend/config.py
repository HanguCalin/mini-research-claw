"""Centralized configuration: model assignments, thresholds, sandbox settings.

All tunables that drive pipeline behavior live here so nodes never hard-code
magic numbers and tests can monkey-patch a single module.

Sourced from `Mini_Research_Claw_Full_Plan.md` §6 (Models per node) and §7
(Thresholds), and from `IMPLEMENTATION_GUIDE.md` §1.7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load `.env` once at import time. `override=True` makes the project's `.env`
# authoritative over inherited shell exports — otherwise a stale
# `export SUPABASE_URL=...` in ~/.bashrc silently shadows the file.
load_dotenv(override=True)

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CACHE_ROOT: Final[Path] = PROJECT_ROOT / ".cache"
PIP_CACHE_DIR: Final[Path] = CACHE_ROOT / "pip"
HF_CACHE_DIR: Final[Path] = CACHE_ROOT / "hf"
SKLEARN_CACHE_DIR: Final[Path] = CACHE_ROOT / "sklearn"


# ─── Secrets (loaded from .env) ──────────────────────────────────────────────
ANTHROPIC_API_KEY: Final[str | None] = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL: Final[str | None] = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY: Final[str | None] = os.getenv("SUPABASE_SERVICE_KEY")


# ─── Model assignments per node ──────────────────────────────────────────────
# Cheap models for high-volume / structured tasks; stronger model for reasoning
# (hypothesis generation, experiment design, ML coding, fact-checking).
@dataclass(frozen=True)
class ModelAssignments:
    kg_extractor: str = "claude-haiku-4-5-20251001"
    hypothesis_generator: str = "claude-sonnet-4-6"
    experiment_designer: str = "claude-sonnet-4-6"
    ml_coder: str = "claude-sonnet-4-6"
    academic_writer: str = "claude-sonnet-4-6"
    critique_fact_checker: str = "claude-sonnet-4-6"
    critique_methodologist: str = "claude-haiku-4-5-20251001"
    critique_formatter: str = "claude-haiku-4-5-20251001"
    kg_dedup: str = "claude-haiku-4-5-20251001"
    latex_repair: str = "claude-haiku-4-5-20251001"


MODELS: Final[ModelAssignments] = ModelAssignments()


# ─── Pipeline thresholds (see plan §7 Configuration) ─────────────────────────
@dataclass(frozen=True)
class Thresholds:
    # Novelty gating (Node 3)
    novelty_threshold: float = 0.35              # min RND for "novel enough"
    prior_art_ceiling: float = 0.90              # max cosine sim to any prior paper
    sbert_dedup_threshold: float = 0.85          # KG entity merge threshold

    # Retry / loop bounds
    max_retrieval_rounds: int = 3                # Node 1 iterative literature loop
    max_code_retries: int = 3                    # Node 4/5 self-healing loop
    max_latex_repair_attempts: int = 5           # Node 9 compile-repair loop

    # arXiv
    arxiv_results_per_round: int = 5
    arxiv_rate_limit_seconds: float = 3.0        # ToS-compliant pacing

    # Claim-ledger No-Paper gate (Node 5b)
    no_paper_weak_fraction: float = 0.50         # >50% weak/unsupported → no_paper

    # pgvector prior-art lookup (Node 3)
    prior_art_top_k: int = 10


THRESHOLDS: Final[Thresholds] = Thresholds()


# ─── Embeddings ──────────────────────────────────────────────────────────────
SBERT_MODEL_NAME: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM: Final[int] = 384


# ─── Docker sandbox settings (Node 5 Executor) ───────────────────────────────
@dataclass(frozen=True)
class SandboxConfig:
    image_tag: str = "auto-mini-claw-sandbox:latest"
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    network_mode: str = "none"                   # strict isolation
    read_only_root: bool = True
    no_new_privileges: bool = True
    workdir: str = "/workspace"
    timeout_seconds: int = 600                   # hard kill after 10 min

    # Read-only mounts populated by the host-side Dependency Resolver.
    pip_mount: str = "/pip_cache"
    hf_mount: str = "/hf_cache"
    sklearn_mount: str = "/sklearn_cache"


SANDBOX: Final[SandboxConfig] = SandboxConfig()


# ─── Supabase Storage layout ─────────────────────────────────────────────────
ARTIFACTS_BUCKET: Final[str] = "artifacts"
PAPERS_TABLE: Final[str] = "papers"
PIPELINE_RUNS_TABLE: Final[str] = "pipeline_runs"


# ─── Validation ──────────────────────────────────────────────────────────────
def assert_env_ready() -> None:
    """Fail fast at startup if required secrets are missing.

    Call this from `backend/main.py` before building the LangGraph DAG so
    the operator sees a clear error rather than a cryptic runtime failure
    deep inside a node.
    """
    missing = [
        name
        for name, value in {
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "SUPABASE_URL": SUPABASE_URL,
            "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy `.env.example` to `.env` and fill in real values."
        )
