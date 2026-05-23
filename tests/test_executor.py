"""Unit tests for backend/agents/executor.py

Tests:
- On success: execution_success=True, logs and metrics captured
- On failure below retry limit: execution_success=False, retry count incremented
- On failure at retry limit: pipeline_status="failed_execution"
- metrics_json defaults to {} when run_sandboxed returns None
"""

from unittest.mock import patch

import pytest

from backend.agents.executor import executor


BASE_STATE = {
    "python_code": "import json\njson.dump({'accuracy': 0.95}, open('metrics.json','w'))",
    "resolved_dependencies": ["scikit-learn", "numpy"],
    "dataset_cache_path": "/tmp/cache",
    "code_retry_count": 0,
}


# ─── Success path ─────────────────────────────────────────────────────────────


@patch("backend.agents.executor.run_sandboxed",
       return_value=(True, "stdout: done", {"accuracy": 0.95}))
def test_success_sets_execution_success_true(mock_run):
    result = executor(BASE_STATE)
    assert result["execution_success"] is True


@patch("backend.agents.executor.run_sandboxed",
       return_value=(True, "stdout: done", {"accuracy": 0.95}))
def test_success_captures_logs(mock_run):
    result = executor(BASE_STATE)
    assert result["execution_logs"] == "stdout: done"


@patch("backend.agents.executor.run_sandboxed",
       return_value=(True, "stdout: done", {"accuracy": 0.95}))
def test_success_captures_metrics(mock_run):
    result = executor(BASE_STATE)
    assert result["metrics_json"] == {"accuracy": 0.95}


@patch("backend.agents.executor.run_sandboxed",
       return_value=(True, "stdout: done", None))
def test_success_defaults_metrics_to_empty_dict(mock_run):
    result = executor(BASE_STATE)
    assert result["metrics_json"] == {}


# ─── Failure below retry limit ────────────────────────────────────────────────


@patch("backend.agents.executor.run_sandboxed",
       return_value=(False, "Error: ModuleNotFoundError", None))
def test_failure_sets_execution_success_false(mock_run):
    result = executor(BASE_STATE)
    assert result["execution_success"] is False


@patch("backend.agents.executor.run_sandboxed",
       return_value=(False, "Error: ModuleNotFoundError", None))
def test_failure_increments_retry_count(mock_run):
    result = executor(BASE_STATE)
    assert result["code_retry_count"] == 1


@patch("backend.agents.executor.run_sandboxed",
       return_value=(False, "Error: crash", None))
def test_failure_below_limit_has_no_failed_status(mock_run):
    result = executor(BASE_STATE)
    assert result.get("pipeline_status") != "failed_execution"


# ─── Failure at retry limit ───────────────────────────────────────────────────


@patch("backend.agents.executor.run_sandboxed",
       return_value=(False, "Error: crash", None))
def test_failure_at_retry_limit_sets_failed_execution(mock_run):
    from backend.config import THRESHOLDS
    state = {**BASE_STATE, "code_retry_count": THRESHOLDS.max_code_retries - 1}
    result = executor(state)
    assert result["pipeline_status"] == "failed_execution"


@patch("backend.agents.executor.run_sandboxed",
       return_value=(False, "Error: crash", None))
def test_failure_at_retry_limit_sets_execution_success_false(mock_run):
    from backend.config import THRESHOLDS
    state = {**BASE_STATE, "code_retry_count": THRESHOLDS.max_code_retries - 1}
    result = executor(state)
    assert result["execution_success"] is False


# ─── run_sandboxed is called with correct args ────────────────────────────────


@patch("backend.agents.executor.run_sandboxed",
       return_value=(True, "ok", {}))
def test_run_sandboxed_called_with_state_values(mock_run):
    executor(BASE_STATE)
    mock_run.assert_called_once_with(
        BASE_STATE["python_code"],
        BASE_STATE["resolved_dependencies"],
        BASE_STATE["dataset_cache_path"],
    )
