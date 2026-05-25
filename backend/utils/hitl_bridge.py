"""Bridge between the LangGraph pipeline thread and the FastAPI request handler.

When ``AUTO_MINI_CLAW_HITL_MODE=api`` the HITL nodes no longer call ``input()``
on stdin. Instead they:

1. Look up the current run's ``run_id`` from a thread-local set by the API.
2. Publish the review payload (everything the CLI's Rich panels would have
   shown) to ``_pending``.
3. Block on a ``threading.Event`` until the API receives a POST and calls
   :func:`submit_decision`, which writes the decision back into ``_pending``
   and sets the event.

The pipeline thread then reads the decision and returns control to LangGraph.
The whole exchange is in-process, so it works fine for a single-worker
deployment (which is what ``docker-compose.yml`` provisions).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal, TypedDict

HITLMode = Literal["cli", "api"]


def current_mode() -> HITLMode:
    """Return ``"api"`` when the API has bound a run id, else ``"cli"``."""
    raw = os.getenv("AUTO_MINI_CLAW_HITL_MODE", "cli").strip().lower()
    return "api" if raw == "api" else "cli"


# ─── Run-id binding (thread-local) ───────────────────────────────────────────
_current = threading.local()


def bind_run_id(run_id: str) -> None:
    """Associate the calling thread with an API ``run_id``.

    Called by ``backend.api._execute_run`` immediately before invoking
    ``run_pipeline()`` so that any HITL gate inside the graph can publish its
    review payload under the correct id.
    """
    _current.run_id = run_id


def clear_run_id() -> None:
    """Drop the thread-local binding (call in a ``finally`` block)."""
    if hasattr(_current, "run_id"):
        del _current.run_id


def get_run_id() -> str | None:
    return getattr(_current, "run_id", None)


# ─── Pending-gate registry ───────────────────────────────────────────────────
class PendingGate(TypedDict):
    run_id: str
    gate_id: Literal["hypothesis", "experiment"]
    payload: dict[str, Any]
    event: threading.Event
    decision: dict[str, Any] | None


_pending: dict[str, PendingGate] = {}
_lock = threading.Lock()


def await_decision(
    gate_id: Literal["hypothesis", "experiment"],
    payload: dict[str, Any],
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Block the pipeline thread until the UI posts an approve/reject.

    Raises ``RuntimeError`` if no run id is bound (i.e. the bridge was called
    outside an API context) or if the wait times out.
    """
    run_id = get_run_id()
    if run_id is None:
        raise RuntimeError(
            "hitl_bridge.await_decision called without a bound run_id — "
            "this code path is only valid when invoked from the FastAPI "
            "executor with AUTO_MINI_CLAW_HITL_MODE=api."
        )

    event = threading.Event()
    entry: PendingGate = {
        "run_id": run_id,
        "gate_id": gate_id,
        "payload": payload,
        "event": event,
        "decision": None,
    }
    with _lock:
        _pending[run_id] = entry

    try:
        if not event.wait(timeout=timeout_seconds):
            raise RuntimeError(
                f"HITL gate {gate_id!r} timed out after {timeout_seconds}s "
                f"waiting for a UI decision (run_id={run_id})"
            )
        decision = entry["decision"]
        assert decision is not None  # set by submit_decision before event
        return decision
    finally:
        with _lock:
            _pending.pop(run_id, None)


def peek_pending(run_id: str) -> dict[str, Any] | None:
    """Return the pending-gate snapshot for ``run_id`` (or ``None``).

    Used by ``GET /api/runs/{run_id}/pending-gate``. Returns a plain dict
    safe to serialise (no ``threading.Event``).
    """
    with _lock:
        entry = _pending.get(run_id)
        if entry is None:
            return None
        return {
            "run_id": entry["run_id"],
            "gate_id": entry["gate_id"],
            "payload": entry["payload"],
        }


def submit_decision(
    run_id: str,
    gate_id: Literal["hypothesis", "experiment"],
    action: Literal["approve", "reject"],
    reason: str = "",
) -> bool:
    """Record an approve/reject and release the waiting pipeline thread.

    Returns ``True`` on success, ``False`` if no matching pending gate is
    found (e.g. the UI raced ahead of the pipeline).
    """
    with _lock:
        entry = _pending.get(run_id)
        if entry is None or entry["gate_id"] != gate_id:
            return False
        entry["decision"] = {"action": action, "reason": reason}
        entry["event"].set()
    return True
