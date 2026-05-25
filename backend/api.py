"""HTTP API for running Auto-Mini-Claw from the frontend.

The CLI remains available via `backend.main`. This module wraps the same
`run_pipeline()` function in a small FastAPI service so the React app can start
and poll research runs without shelling out.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import ARTIFACTS_BUCKET
from backend.utils.supabase_client import get_supabase

from backend.graph import run_pipeline
from backend.utils import hitl_bridge, run_overrides
from backend.utils.run_overrides import RunOverrides

logger = logging.getLogger(__name__)

RunStatus = Literal["queued", "running", "success", "failed"]


class CreateRunRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)
    # Optional per-run tunables. ``None`` falls through to the defaults in
    # ``backend.config`` — sending nothing here keeps the original behaviour.
    max_code_retries: int | None = Field(default=None, ge=0, le=10)
    arxiv_results_per_round: int | None = Field(default=None, ge=1, le=50)
    model_override: str | None = Field(default=None, max_length=120)


class RunRecord(BaseModel):
    client_run_id: str
    topic: str
    status: RunStatus
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None
    overrides: dict[str, Any] | None = None


app = FastAPI(title="Auto-Mini-Claw API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runs: dict[str, RunRecord] = {}
_runs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("AUTO_MINI_CLAW_API_WORKERS", "1")))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _save(record: RunRecord) -> None:
    with _runs_lock:
        _runs[record.client_run_id] = record


def _get(run_id: str) -> RunRecord | None:
    with _runs_lock:
        return _runs.get(run_id)


def _execute_run(run_id: str, topic: str, overrides: RunOverrides | None) -> None:
    record = _get(run_id)
    if not record:
        return

    _save(record.model_copy(update={"status": "running", "updated_at": _now()}))
    logger.info(
        "API run started client_run_id=%s topic=%r overrides=%s",
        run_id, topic, overrides,
    )

    hitl_bridge.bind_run_id(run_id)
    run_overrides.bind(overrides)
    try:
        result = dict(run_pipeline(topic, run_id=run_id))
        pipeline_status = str(result.get("pipeline_status", "unknown"))
        status: RunStatus = "success" if pipeline_status == "success" else "failed"
        _save(
            RunRecord(
                client_run_id=run_id,
                topic=topic,
                status=status,
                created_at=record.created_at,
                updated_at=_now(),
                result=result,
                error=None if status == "success" else pipeline_status,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("API run crashed client_run_id=%s: %s", run_id, exc)
        _save(
            record.model_copy(
                update={
                    "status": "failed",
                    "updated_at": _now(),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
        )
    finally:
        hitl_bridge.clear_run_id()
        run_overrides.clear()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunRecord, status_code=202)
def create_run(payload: CreateRunRequest) -> RunRecord:
    run_id = str(uuid.uuid4())
    overrides: RunOverrides | None = None
    if (
        payload.max_code_retries is not None
        or payload.arxiv_results_per_round is not None
        or payload.model_override
    ):
        overrides = RunOverrides(
            max_code_retries=payload.max_code_retries,
            arxiv_results_per_round=payload.arxiv_results_per_round,
            model_override=(payload.model_override or None),
        )
    record = RunRecord(
        client_run_id=run_id,
        topic=payload.topic.strip(),
        status="queued",
        created_at=_now(),
        updated_at=_now(),
        overrides=(
            {
                "max_code_retries": overrides.max_code_retries,
                "arxiv_results_per_round": overrides.arxiv_results_per_round,
                "model_override": overrides.model_override,
            }
            if overrides
            else None
        ),
    )
    _save(record)
    _executor.submit(_execute_run, run_id, record.topic, overrides)
    return record


@app.get("/api/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    record = _get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record


# ─── HITL gate endpoints ─────────────────────────────────────────────────────
class GateDecisionRequest(BaseModel):
    gate_id: Literal["hypothesis", "experiment"]
    action: Literal["approve", "reject"]
    reason: str = Field(default="", max_length=2000)


@app.get("/api/runs/{run_id}/pending-gate")
def pending_gate(run_id: str) -> dict[str, Any]:
    """Return the pending HITL gate payload for this run, or 204 if none."""
    if _get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    snapshot = hitl_bridge.peek_pending(run_id)
    if snapshot is None:
        # No gate currently waiting — the UI will poll again.
        return {"pending": False}
    return {"pending": True, **snapshot}


# ─── Artifact download endpoints ─────────────────────────────────────────────
# Whitelist of filenames the uploader produces. Used to reject path-traversal
# attempts before we ever ask Supabase Storage for a blob.
_ALLOWED_ARTIFACTS: frozenset[str] = frozenset({
    "metrics.json",
    "claim_ledger.json",
    "debate_log.json",
    "draft.tex",
    "references.bib",
    "python_code.py",
    "execution_logs.txt",
    "hypothesis.txt",
    "experiment_spec.json",
    "final_paper.pdf",
    "failure_report.json",
})

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".tex": "text/x-tex",
    ".bib": "text/x-bibtex",
    ".py": "text/x-python",
    ".txt": "text/plain",
}


@app.get("/api/runs/{run_id}/artifacts")
def list_artifacts(run_id: str) -> dict[str, Any]:
    """List the artifact filenames currently in Supabase Storage for this run.

    Returns ``{"files": ["failure_report.json", ...]}`` ordered by name. The
    UI uses this to decide which download buttons to render — a failed run
    typically has the diagnostic set but no ``final_paper.pdf``.
    """
    if _get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    sb = get_supabase()
    try:
        entries = sb.storage.from_(ARTIFACTS_BUCKET).list(run_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Storage list failed: {exc}") from exc
    names = sorted(
        e["name"] for e in entries
        if e.get("name") and e["name"] in _ALLOWED_ARTIFACTS
    )
    return {"run_id": run_id, "files": names}


@app.get("/api/runs/{run_id}/artifacts/{filename}")
def download_artifact(run_id: str, filename: str) -> Response:
    """Stream an individual artifact from Supabase Storage as a download."""
    if filename not in _ALLOWED_ARTIFACTS:
        raise HTTPException(status_code=400, detail=f"Filename {filename!r} not allowed")
    if _get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    sb = get_supabase()
    try:
        blob = sb.storage.from_(ARTIFACTS_BUCKET).download(f"{run_id}/{filename}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=404,
            detail=f"Artifact {filename!r} not found in storage for this run",
        ) from exc
    suffix = filename[filename.rfind("."):].lower() if "." in filename else ""
    return Response(
        content=blob,
        media_type=_CONTENT_TYPES.get(suffix, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/runs/{run_id}/gate-decision")
def gate_decision(run_id: str, payload: GateDecisionRequest) -> dict[str, Any]:
    """Record the operator's approve/reject and release the pipeline thread."""
    if _get(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    accepted = hitl_bridge.submit_decision(
        run_id=run_id,
        gate_id=payload.gate_id,
        action=payload.action,
        reason=payload.reason,
    )
    if not accepted:
        raise HTTPException(
            status_code=409,
            detail=(
                f"No pending {payload.gate_id!r} gate for this run — "
                "the pipeline may have moved on already."
            ),
        )
    return {"accepted": True}
