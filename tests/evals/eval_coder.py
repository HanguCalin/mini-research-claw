"""Eval: ML Coder quality — 6 checks from §7.3.

Runs the real LLM (requires ANTHROPIC_API_KEY). Checks:
  1. Syntax Validity        — ast.parse() succeeds on 10/10 runs
  2. Debug Instrumentation  — >= 3 strategic print() statements
  3. ML Rigor               — train/test split, random seed, cross-validation
  4. ExperimentSpec Compliance — uses correct dataset, metrics, variables
  5. Static Import Compliance  — no importlib/exec/eval/__import__/subprocess
  6. Pre-Compiled Wheels Only  — all imports come from the allowlist

Pass criteria per the plan:
  1. No SyntaxError on 10/10
  2. >= 3 debug prints per script
  3. All 3 rigor practices present on 10/10
  4. 100% spec compliance on 10/10
  5. 0 violations on 10/10
  6. 100% allowlist compliance on 10/10
"""

import ast
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.ml_coder import ml_coder
from backend.state import AutoResearchState

# ─── Allowlist (must match ml_coder system prompt) ────────────────────────────

WHEEL_ALLOWLIST = {
    "sklearn", "scikit_learn",
    "transformers",
    "torch", "torchvision", "torchaudio",
    "pandas", "pd",
    "numpy", "np",
    "scipy",
    "datasets",
    "huggingface_hub",
    "matplotlib", "plt",
    "seaborn", "sns",
    "json", "os", "sys", "math", "random", "time", "datetime",
    "collections", "itertools", "functools", "typing",
    "pathlib", "io", "copy", "abc", "dataclasses", "enum",
}

FORBIDDEN_PATTERNS = [
    r"\bimportlib\b",
    r"\bimportlib\.import_module\b",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\b__import__\s*\(",
    r"\bsubprocess\b",
]

# ─── Gold ExperimentSpec ──────────────────────────────────────────────────────

SPEC = {
    "independent_var": "model_type",
    "dependent_var": "accuracy",
    "control_description": "Logistic Regression baseline",
    "dataset_id": "iris",
    "evaluation_metrics": ["accuracy", "f1"],
    "expected_outcome": "RF achieves higher accuracy than Logistic Regression on iris",
}

HYPOTHESIS = (
    "Random Forest outperforms Logistic Regression on the iris dataset "
    "when using 5-fold cross-validation, with accuracy and F1 as metrics."
)

BASE_STATE: AutoResearchState = {
    "topic": "RF vs Logistic Regression",
    "run_id": "eval_coder",
    "retrieval_round": 1,
    "code_retry_count": 0,
    "hypothesis": HYPOTHESIS,
    "experiment_spec": SPEC,
    "kg_entities": [],
    "kg_edges": [],
}

# ─── Checkers ─────────────────────────────────────────────────────────────────

def check_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, str(e)


def check_debug_prints(code: str) -> tuple[bool, str]:
    count = len(re.findall(r"\bprint\s*\(", code))
    return count >= 3, f"{count} print() calls found"


def check_ml_rigor(code: str) -> tuple[bool, str]:
    has_split = bool(re.search(r"train_test_split|X_train|X_test", code))
    has_seed = bool(re.search(r"random_state\s*=|random\.seed\s*\(|np\.random\.seed\s*\(", code))
    has_cv = bool(re.search(r"cross_val|cross_validate|KFold|StratifiedKFold", code))
    missing = [k for k, v in [("train/test split", has_split), ("random seed", has_seed), ("cross-validation", has_cv)] if not v]
    return len(missing) == 0, f"missing: {missing}" if missing else "all 3 present"


def check_spec_compliance(code: str, spec: dict) -> tuple[bool, str]:
    issues = []
    if spec["dataset_id"] not in code:
        issues.append(f"dataset '{spec['dataset_id']}' not referenced")
    for metric in spec["evaluation_metrics"]:
        if metric not in code:
            issues.append(f"metric '{metric}' not referenced")
    if spec["independent_var"] not in code and spec["dependent_var"] not in code:
        issues.append("neither IV nor DV variable name referenced")
    return len(issues) == 0, "; ".join(issues) if issues else "compliant"


def check_static_imports(code: str) -> tuple[bool, str]:
    violations = []
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            violations.append(pattern)
    return len(violations) == 0, f"violations: {violations}" if violations else "clean"


def check_allowlist(code: str) -> tuple[bool, str]:
    tree = ast.parse(code)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in WHEEL_ALLOWLIST:
                    bad.append(root)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root not in WHEEL_ALLOWLIST:
                    bad.append(root)
    return len(bad) == 0, f"non-allowlisted: {list(set(bad))}" if bad else "all allowed"


# ─── Runner ───────────────────────────────────────────────────────────────────

CHECKS = [
    ("Syntax Validity",           check_syntax),
    ("Debug Instrumentation",     check_debug_prints),
    ("ML Rigor",                  check_ml_rigor),
    ("ExperimentSpec Compliance", lambda c: check_spec_compliance(c, SPEC)),
    ("Static Import Compliance",  check_static_imports),
    ("Pre-Compiled Wheels Only",  check_allowlist),
]

RUNS = 3  # plan says 10/10; use 3 by default to save tokens, override with EVAL_RUNS env var

def run_eval():
    print("Starting ML Coder Evaluation (requires ANTHROPIC_API_KEY)...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    n_runs = int(os.getenv("EVAL_RUNS", str(RUNS)))
    results: dict[str, list[bool]] = {name: [] for name, _ in CHECKS}

    for run_idx in range(1, n_runs + 1):
        print(f"\n--- Run {run_idx}/{n_runs} ---")
        state = dict(BASE_STATE)
        try:
            output = ml_coder(state)
            code = output.get("python_code", "")
        except Exception as e:
            print(f"  Agent call failed: {e}")
            for name, _ in CHECKS:
                results[name].append(False)
            continue

        if not code:
            print("  No code returned.")
            for name, _ in CHECKS:
                results[name].append(False)
            continue

        for name, checker in CHECKS:
            ok, detail = checker(code)
            results[name].append(ok)
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name}: {detail}")

    print("\n=== Summary ===")
    all_passed = True
    for name, passes in results.items():
        pct = sum(passes) / len(passes) * 100
        status = "PASS" if all(passes) else "WARN"
        if not all(passes):
            all_passed = False
        print(f"  [{status}] {name}: {sum(passes)}/{len(passes)} ({pct:.0f}%)")

    if all_passed:
        print("\nEvaluation PASSED — all checks 100% on all runs")
    else:
        print("\nEvaluation PARTIAL — review failing checks above")


if __name__ == "__main__":
    run_eval()
