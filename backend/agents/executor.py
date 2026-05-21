"""Node 5: Executor Sandbox.

Type: Non-AI (pure Python + Docker SDK).
Runs the generated code inside a hardened Docker container with
--network=none, --read-only, --security-opt=no-new-privileges.
Drives the retry loop based on exit codes.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.config import THRESHOLDS
from backend.state import AutoResearchState
from backend.utils.docker_utils import run_sandboxed

logger = logging.getLogger(__name__)


def executor(state: AutoResearchState) -> dict[str, Any]:
    """Execute experiment code in sandbox, capture results or trigger retry."""
    code = state.get("python_code", "")
    resolved_deps = state.get("resolved_dependencies", [])
    cache_path = state.get("dataset_cache_path", "")
    retry_count = state.get("code_retry_count", 0)

    success, logs, metrics = run_sandboxed(code, resolved_deps, cache_path)

    if success:
        logger.info("Execution succeeded on attempt %d", retry_count + 1)
        return {
            "execution_success": True,
            "execution_logs": logs,
            "metrics_json": metrics or {},
            "code_retry_count": retry_count,
        }

    retry_count += 1
    logger.warning("Execution failed (attempt %d/%d)",
                    retry_count, THRESHOLDS.max_code_retries)

    if retry_count >= THRESHOLDS.max_code_retries:
        return {
            "execution_success": False,
            "execution_logs": logs,
            "code_retry_count": retry_count,
            "pipeline_status": "failed_execution",
        }

    return {
        "execution_success": False,
        "execution_logs": logs,
        "code_retry_count": retry_count,
    }
