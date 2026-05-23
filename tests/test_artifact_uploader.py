"""Unit tests for backend/utils/artifact_uploader.py

Tests:
- create_run inserts a row and returns a UUID run_id
- upload_artifacts uploads expected files and returns url mapping
- upload_artifacts skips None fields (no upload for missing data)
- upload_artifacts adds failure_report.json on non-success status
- upload_artifacts does NOT add failure_report.json on success status
- finalize_run updates pipeline_runs row with status and completed_at
- finalize_run is a no-op when run_id is missing
- _json_dump returns None for None input, valid JSON otherwise
"""

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from backend.utils.artifact_uploader import (
    _json_dump,
    create_run,
    finalize_run,
    upload_artifacts,
)


def _make_mock_supabase():
    sb = MagicMock()
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock()
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()
    sb.storage.from_.return_value.upload.return_value = MagicMock()
    return sb


BASE_STATE = {
    "run_id": "test-run-123",
    "topic": "RF vs XGBoost",
    "pipeline_status": "success",
    "hypothesis": "RF wins.",
    "latex_draft": r"\documentclass{article}\begin{document}Hello\end{document}",
    "bibtex_source": "@article{ref1}",
    "metrics_json": {"accuracy": 0.95},
    "claim_ledger": [{"claim_id": "c1", "claim_text": "RF wins."}],
    "debate_log": [],
    "python_code": "print('hello')",
    "execution_logs": "stdout: hello",
    "experiment_spec": {"dataset_id": "iris"},
    "final_pdf_path": None,
}


# ─── _json_dump ───────────────────────────────────────────────────────────────


def test_json_dump_returns_none_for_none():
    assert _json_dump(None) is None


def test_json_dump_returns_valid_json_string():
    result = _json_dump({"key": "value"})
    import json
    assert json.loads(result) == {"key": "value"}


def test_json_dump_handles_list():
    result = _json_dump([1, 2, 3])
    import json
    assert json.loads(result) == [1, 2, 3]


# ─── create_run ───────────────────────────────────────────────────────────────


@patch("backend.utils.artifact_uploader.get_supabase")
def test_create_run_returns_string_id(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    run_id = create_run("test topic")
    assert isinstance(run_id, str)
    assert len(run_id) == 36  # UUID format


@patch("backend.utils.artifact_uploader.get_supabase")
def test_create_run_inserts_with_running_status(mock_get_sb):
    sb = _make_mock_supabase()
    mock_get_sb.return_value = sb
    run_id = create_run("test topic")
    insert_call = sb.table.return_value.insert.call_args[0][0]
    assert insert_call["status"] == "running"
    assert insert_call["topic"] == "test topic"


# ─── upload_artifacts ─────────────────────────────────────────────────────────


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_returns_dict(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    result = upload_artifacts(BASE_STATE)
    assert isinstance(result, dict)


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_includes_latex_draft(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    result = upload_artifacts(BASE_STATE)
    assert "draft.tex" in result


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_includes_metrics_json(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    result = upload_artifacts(BASE_STATE)
    assert "metrics.json" in result


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_skips_none_fields(mock_get_sb):
    sb = _make_mock_supabase()
    mock_get_sb.return_value = sb
    state = {**BASE_STATE, "latex_draft": None}
    result = upload_artifacts(state)
    assert "draft.tex" not in result


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_adds_failure_report_on_failed_status(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    state = {**BASE_STATE, "pipeline_status": "failed_execution"}
    result = upload_artifacts(state)
    assert "failure_report.json" in result


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_no_failure_report_on_success(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    result = upload_artifacts(BASE_STATE)
    assert "failure_report.json" not in result


@patch("backend.utils.artifact_uploader.get_supabase")
def test_upload_paths_include_run_id_prefix(mock_get_sb):
    mock_get_sb.return_value = _make_mock_supabase()
    result = upload_artifacts(BASE_STATE)
    for path in result.values():
        assert path.startswith("test-run-123/")


# ─── finalize_run ─────────────────────────────────────────────────────────────


@patch("backend.utils.artifact_uploader.get_supabase")
def test_finalize_run_updates_status(mock_get_sb):
    sb = _make_mock_supabase()
    mock_get_sb.return_value = sb
    finalize_run(BASE_STATE)
    update_call = sb.table.return_value.update.call_args[0][0]
    assert update_call["status"] == "success"


@patch("backend.utils.artifact_uploader.get_supabase")
def test_finalize_run_sets_artifact_path(mock_get_sb):
    sb = _make_mock_supabase()
    mock_get_sb.return_value = sb
    finalize_run(BASE_STATE)
    update_call = sb.table.return_value.update.call_args[0][0]
    assert update_call["artifact_path"] == "artifacts/test-run-123/"


@patch("backend.utils.artifact_uploader.get_supabase")
def test_finalize_run_noop_when_no_run_id(mock_get_sb):
    sb = _make_mock_supabase()
    mock_get_sb.return_value = sb
    finalize_run({})
    sb.table.assert_not_called()
