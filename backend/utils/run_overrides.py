"""Per-run overrides for the model assignments and tunable thresholds.

The defaults in :mod:`backend.config` (``MODELS``, ``THRESHOLDS``) are frozen
dataclasses that the agents import at module load time, so they cannot be
mutated. When the API receives a request that wants to customise behaviour
for a single run, we record the overrides in a thread-local registry and the
agents read through these helpers instead of touching the dataclasses
directly.

CLI runs and any request that doesn't supply overrides just see the defaults.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from backend.config import MODELS, THRESHOLDS, ModelAssignments

NODE_NAMES = tuple(f.name for f in ModelAssignments.__dataclass_fields__.values())


@dataclass(frozen=True)
class RunOverrides:
    """A single run's optional overrides.

    ``model_override`` (when set) is applied to *every* AI node — useful for
    switching the whole pipeline to a single model for a test run. Any field
    set to ``None`` falls through to the corresponding default.
    """

    max_code_retries: Optional[int] = None
    arxiv_results_per_round: Optional[int] = None
    model_override: Optional[str] = None


# ─── Thread-local binding ────────────────────────────────────────────────────
_current = threading.local()


def bind(overrides: Optional[RunOverrides]) -> None:
    """Associate the calling thread with a set of overrides (or clear them)."""
    if overrides is None:
        clear()
        return
    _current.overrides = overrides


def clear() -> None:
    if hasattr(_current, "overrides"):
        del _current.overrides


def current() -> Optional[RunOverrides]:
    return getattr(_current, "overrides", None)


# ─── Read helpers used by the agents and graph router ───────────────────────
def effective_max_code_retries() -> int:
    o = current()
    if o is not None and o.max_code_retries is not None:
        return o.max_code_retries
    return THRESHOLDS.max_code_retries


def effective_arxiv_results_per_round() -> int:
    o = current()
    if o is not None and o.arxiv_results_per_round is not None:
        return o.arxiv_results_per_round
    return THRESHOLDS.arxiv_results_per_round


def effective_model_for(node_name: str) -> str:
    """Return the Anthropic model id to use for ``node_name``.

    Falls back to the default mapping in :data:`backend.config.MODELS`. When a
    run-wide ``model_override`` is bound it takes precedence for every node so
    that "swap the whole pipeline to model X" is a one-line API request.
    """
    o = current()
    if o is not None and o.model_override:
        return o.model_override
    if node_name not in NODE_NAMES:
        raise KeyError(
            f"Unknown model slot {node_name!r}; valid slots are {NODE_NAMES}"
        )
    return getattr(MODELS, node_name)
