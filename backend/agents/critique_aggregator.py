"""Node 8: Critique Aggregator.

Type: Non-AI (pure Python).
Merges linter warnings (which bypassed debate) with debate-surviving
critiques into a single feedback list routed to Node 6 for revision.
"""

from __future__ import annotations

from typing import Any

from backend.state import AutoResearchState


def critique_aggregator(state: AutoResearchState) -> dict[str, Any]:
    """Merge linter + debate-surviving critiques and trigger revision pass."""
    critique_warnings = state.get("critique_warnings", [])
    surviving = state.get("surviving_critiques", [])

    linter_warnings = [
        w for w in critique_warnings if w.get("source") == "linter"
    ]

    seen_messages: set[str] = set()
    merged: list[dict[str, Any]] = []

    for w in linter_warnings:
        msg = w.get("message", "")
        if msg not in seen_messages:
            merged.append(w)
            seen_messages.add(msg)

    for c in surviving:
        msg = c.get("critique", c.get("message", ""))
        if msg not in seen_messages:
            merged.append(c)
            seen_messages.add(msg)

    return {
        "critique_warnings": merged,
        "revision_pass_done": True,
    }
