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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.graph import run_pipeline

logger = logging.getLogger(__name__)

RunStatus = Literal["queued", "running", "success", "failed"]


class CreateRunRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)


class RunRecord(BaseModel):
    client_run_id: str
    topic: str
    status: RunStatus
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


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


def _execute_run(run_id: str, topic: str) -> None:
    record = _get(run_id)
    if not record:
        return

    _save(record.model_copy(update={"status": "running", "updated_at": _now()}))
    logger.info("API run started client_run_id=%s topic=%r", run_id, topic)

    try:
        result = dict(run_pipeline(topic))
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/runs", response_model=RunRecord, status_code=202)
def create_run(payload: CreateRunRequest) -> RunRecord:
    run_id = str(uuid.uuid4())
    record = RunRecord(
        client_run_id=run_id,
        topic=payload.topic.strip(),
        status="queued",
        created_at=_now(),
        updated_at=_now(),
    )
    _save(record)
    _executor.submit(_execute_run, run_id, record.topic)
    return record


@app.get("/api/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str) -> RunRecord:
    record = _get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return record
