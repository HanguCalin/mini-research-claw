"""Eval: State Pruning Field Isolation.

Verifies that build_scoped_view() returns exactly the allowed fields for each
AI node per NODE_SCOPE_CONFIG, and that no prohibited fields leak through.
No API key required — pure deterministic logic.

Pass criteria: 100% correct field scoping across all AI nodes.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.state_pruning import NODE_SCOPE_CONFIG, build_scoped_view

# ─── Full state fixture ───────────────────────────────────────────────────────

FULL_STATE = {
    "topic": "Compare RF vs XGBoost",
    "arxiv_papers_full_text": [{"arxiv_id": "2401.00001", "title": "Paper A"}],
    "retrieval_round": 1,
    "kg_entities": [{"id": "e1", "canonical_name": "Random Forest", "entity_type": "model", "aliases": []}],
    "kg_edges": [{"source_id": "e1", "target_id": "e2", "relation": "outperforms", "polarity": "supports",
                  "context_condition": "", "confidence": 0.95, "provenance": "p1/results"}],
    "hypothesis": "RF outperforms XGBoost on small tabular datasets",
    "incremental_delta": "Focuses on datasets with < 10k samples",
    "novelty_passed": True,
    "experiment_spec": {
        "independent_var": "model", "dependent_var": "accuracy",
        "control_description": "same split", "dataset_id": "iris",
        "evaluation_metrics": ["accuracy"], "expected_outcome": "RF >= XGBoost",
    },
    "python_code": "import sklearn\nprint(1)",
    "execution_logs": "stdout: 0.96\nstderr: ",
    "metrics_json": {"accuracy": 0.96},
    "claim_ledger": [{"claim_id": "c1", "claim_text": "RF wins", "supporting_kg_edges": [],
                      "contradicting_kg_edges": [], "evidence_strength": "strong"}],
    "latex_draft": r"\section{Introduction} Hello.",
    "bibtex_source": "@article{ref1, title={A}}",
    "critique_warnings": [],
    "pipeline_status": "running",
    "run_id": "test-run-001",
}

# Fields that are SENSITIVE and must NOT appear in certain node views
PROHIBITED = {
    "kg_extractor": {"kg_entities", "kg_edges", "hypothesis", "python_code", "execution_logs",
                     "metrics_json", "claim_ledger", "latex_draft"},
    "hypothesis_generator": {"arxiv_papers_full_text", "python_code", "execution_logs",
                             "metrics_json", "claim_ledger", "latex_draft"},
    "experiment_designer": {"arxiv_papers_full_text", "python_code", "execution_logs",
                            "metrics_json", "claim_ledger", "latex_draft"},
    "ml_coder": {"arxiv_papers_full_text", "kg_entities", "kg_edges",
                 "execution_logs", "claim_ledger", "latex_draft"},
    "ml_coder_retry": {"arxiv_papers_full_text", "kg_entities", "kg_edges",
                       "claim_ledger", "latex_draft"},
    "academic_writer": {"arxiv_papers_full_text", "kg_entities", "kg_edges",
                        "execution_logs", "python_code"},
    "critique_panel": {"arxiv_papers_full_text", "kg_entities", "kg_edges",
                       "execution_logs", "python_code"},
}

# ─── Eval Logic ───────────────────────────────────────────────────────────────

def run_eval():
    print("Starting State Pruning Field Isolation Evaluation (no API key required)...")
    total = 0
    passed = 0

    # 1. Each AI node's view contains exactly the allowed fields
    for node_name, allowed in NODE_SCOPE_CONFIG.items():
        view = build_scoped_view(FULL_STATE, node_name)
        expected_keys = {k for k in allowed if k in FULL_STATE}
        got_keys = set(view.keys())

        total += 1
        if got_keys == expected_keys:
            passed += 1
            print(f"  [OK] {node_name}: exact field match {sorted(got_keys)}")
        else:
            extra = got_keys - expected_keys
            missing = expected_keys - got_keys
            print(f"  [FAIL] {node_name}: extra={extra}, missing={missing}")

    # 2. Prohibited fields must not appear in any node's view
    for node_name, prohibited_fields in PROHIBITED.items():
        view = build_scoped_view(FULL_STATE, node_name)
        leaked = prohibited_fields.intersection(view.keys())
        total += 1
        if not leaked:
            passed += 1
            print(f"  [OK] {node_name} prohibited fields: none leaked")
        else:
            print(f"  [FAIL] {node_name} leaked prohibited fields: {leaked}")

    # 3. Original state must not be mutated
    original_len = len(FULL_STATE)
    for node_name in NODE_SCOPE_CONFIG:
        _ = build_scoped_view(FULL_STATE, node_name)
    total += 1
    if len(FULL_STATE) == original_len:
        passed += 1
        print(f"  [OK] Original state not mutated after all scoped views")
    else:
        print(f"  [FAIL] Original state was mutated!")

    # 4. Unknown node (non-AI) returns full state copy
    non_ai_view = build_scoped_view(FULL_STATE, "dependency_resolver")
    total += 1
    if set(non_ai_view.keys()) == set(FULL_STATE.keys()):
        passed += 1
        print(f"  [OK] Non-AI node gets full state copy")
    else:
        print(f"  [FAIL] Non-AI node view missing keys: {set(FULL_STATE.keys()) - set(non_ai_view.keys())}")

    pct = passed / total * 100
    print(f"\n--- Results ---")
    print(f"Passed: {passed}/{total} ({pct:.0f}%)")

    if passed == total:
        print("\nEvaluation PASSED (100% correct scoping)")
    else:
        print(f"\nEvaluation FAILED ({passed}/{total} — threshold is 100%)")


if __name__ == "__main__":
    run_eval()
