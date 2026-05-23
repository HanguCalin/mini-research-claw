"""Eval: Experiment Designer completeness — §7.3.

Runs the real LLM (requires ANTHROPIC_API_KEY). Checks:
  All 6 required ExperimentSpec fields present and non-empty on every run.

Pass criteria: all 6 fields present on 10/10 runs (default: 3 runs).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.experiment_designer import experiment_designer, REQUIRED_FIELDS
from backend.state import AutoResearchState

# ─── Gold Input ───────────────────────────────────────────────────────────────

HYPOTHESIS = (
    "Random Forest outperforms Logistic Regression on the iris dataset "
    "when using 5-fold cross-validation, achieving higher accuracy and F1."
)

KG_ENTITIES = [
    {"id": "e1", "canonical_name": "Random Forest", "aliases": ["RF"], "entity_type": "model"},
    {"id": "e2", "canonical_name": "Logistic Regression", "aliases": ["LR"], "entity_type": "model"},
    {"id": "e3", "canonical_name": "iris", "aliases": [], "entity_type": "dataset"},
]

KG_EDGES = [
    {
        "id": "ed1", "source_id": "e1", "target_id": "e2",
        "relation": "outperforms", "polarity": "supports",
        "context_condition": "when n < 500", "confidence": 0.9,
        "provenance": "paper1/results",
    }
]

BASE_STATE: AutoResearchState = {
    "topic": "RF vs Logistic Regression",
    "run_id": "eval_designer",
    "retrieval_round": 1,
    "code_retry_count": 0,
    "hypothesis": HYPOTHESIS,
    "kg_entities": KG_ENTITIES,
    "kg_edges": KG_EDGES,
    "hitl_approved": True,
}

# ─── Runner ───────────────────────────────────────────────────────────────────

RUNS = int(os.getenv("EVAL_RUNS", "3"))


def _check_spec(spec: dict) -> tuple[bool, list[str]]:
    missing = [f for f in REQUIRED_FIELDS if not spec.get(f)]
    empty_metrics = (
        isinstance(spec.get("evaluation_metrics"), list)
        and len(spec["evaluation_metrics"]) == 0
    )
    if empty_metrics:
        missing.append("evaluation_metrics (empty list)")
    return len(missing) == 0, missing


def run_eval():
    print("Starting Experiment Designer Evaluation (requires ANTHROPIC_API_KEY)...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    passed = 0

    for run_idx in range(1, RUNS + 1):
        print(f"\n--- Run {run_idx}/{RUNS} ---")
        state = dict(BASE_STATE)
        try:
            output = experiment_designer(state)
            spec = output.get("experiment_spec", {})
        except Exception as e:
            print(f"  Agent call failed: {e}")
            continue

        ok, missing = _check_spec(spec)
        if ok:
            passed += 1
            print(f"  [OK] All 6 fields present")
            print(f"       dataset_id={spec.get('dataset_id')!r}  "
                  f"metrics={spec.get('evaluation_metrics')}")
        else:
            print(f"  [FAIL] Missing fields: {missing}")
            print(f"         Raw spec: {spec}")

    print(f"\n=== Summary ===")
    pct = passed / RUNS * 100
    print(f"  Passed: {passed}/{RUNS} ({pct:.0f}%)")

    if passed == RUNS:
        print("\nEvaluation PASSED — all 6 fields present on every run")
    else:
        print(f"\nEvaluation FAILED — {RUNS - passed} run(s) had incomplete specs")


if __name__ == "__main__":
    run_eval()
