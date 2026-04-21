"""Artifact uploader — persists pipeline outputs to Supabase Storage.

Runs at every terminal state (success, failed_latex, no_paper,
failed_novelty, failed_hitl, failed_execution).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import ARTIFACTS_BUCKET, PIPELINE_RUNS_TABLE
from backend.state import AutoResearchState
from backend.utils.supabase_client import get_supabase


def create_run(topic: str) -> str:
    """Insert a new pipeline_runs row with status='running'. Returns run_id."""
    run_id = str(uuid.uuid4())
    sb = get_supabase()
    sb.table(PIPELINE_RUNS_TABLE).insert({
        "id": run_id,
        "topic": topic,
        "status": "running",
    }).execute()
    return run_id


def upload_artifacts(state: AutoResearchState) -> dict[str, str]:
    """Upload all available artifacts to ``artifacts/{run_id}/``.

    Returns a filename → public-URL mapping for ``state["artifact_urls"]``.
    """
    sb = get_supabase()
    run_id = state.get("run_id", "unknown")
    prefix = f"{run_id}/"
    urls: dict[str, str] = {}

    artifacts: list[tuple[str, str | None]] = [
        ("metrics.json", _json_dump(state.get("metrics_json"))),
        ("claim_ledger.json", _json_dump(state.get("claim_ledger"))),
        ("debate_log.json", _json_dump(state.get("debate_log"))),
        ("draft.tex", state.get("latex_draft")),
        ("references.bib", state.get("bibtex_source")),
    ]

    pdf_path = state.get("final_pdf_path")
    if pdf_path and Path(pdf_path).exists():
        pdf_bytes = Path(pdf_path).read_bytes()
        dest = f"{prefix}final_paper.pdf"
        sb.storage.from_(ARTIFACTS_BUCKET).upload(
            dest, pdf_bytes, {"content-type": "application/pdf"},
        )
        urls["final_paper.pdf"] = dest

    status = state.get("pipeline_status", "unknown")
    if status not in ("success",):
        report = {
            "status": status,
            "run_id": run_id,
            "logs": state.get("logs", []),
        }
        artifacts.append(("failure_report.json", json.dumps(report, indent=2)))

    for filename, content in artifacts:
        if content is None:
            continue
        dest = f"{prefix}{filename}"
        content_bytes = content.encode("utf-8") if isinstance(content, str) else content
        sb.storage.from_(ARTIFACTS_BUCKET).upload(
            dest, content_bytes, {"content-type": "application/octet-stream"},
        )
        urls[filename] = dest

    return urls


def finalize_run(state: AutoResearchState) -> None:
    """Update the pipeline_runs row with final status and artifact path."""
    sb = get_supabase()
    run_id = state.get("run_id")
    if not run_id:
        return

    sb.table(PIPELINE_RUNS_TABLE).update({
        "status": state.get("pipeline_status", "unknown"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": f"artifacts/{run_id}/",
        "metadata": {
            "total_api_calls": state.get("total_api_calls", 0),
            "total_tokens_used": state.get("total_tokens_used", 0),
        },
    }).eq("id", run_id).execute()


def _json_dump(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, indent=2, default=str)
