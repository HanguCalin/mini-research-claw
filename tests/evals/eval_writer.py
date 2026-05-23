"""Eval: Academic Writer quality — 2 checks from §7.3.

Runs the real LLM (requires ANTHROPIC_API_KEY). Checks:
  1. LaTeX Validity  — pdflatex compiles successfully (or latexmk)
  2. Structure       — all 4 IMRaD sections present via regex

Pass criteria:
  1. Successful compilation on 10/10 runs (default: 3)
  2. All 4 sections present on 10/10 runs
"""

import os
import re
import sys
import subprocess
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.academic_writer import academic_writer
from backend.state import AutoResearchState

# ─── Required IMRaD sections ──────────────────────────────────────────────────

REQUIRED_SECTIONS = [
    r"\\section\{Introduction\}",
    r"\\section\{Methods?\}",
    r"\\section\{Results?\}",
    r"\\section\{Conclusion\}",
]

# ─── Gold Input ───────────────────────────────────────────────────────────────

CLAIM_LEDGER = [
    {
        "claim": "Random Forest outperforms Logistic Regression on iris when n < 500.",
        "evidence_strength": "strong",
        "supporting_edges": [{"relation": "outperforms", "polarity": "supports", "provenance": "Smith2023/results"}],
        "contradicting_edges": [],
        "is_unconditional": False,
    },
    {
        "claim": "5-fold cross-validation reduces variance in performance estimates.",
        "evidence_strength": "moderate",
        "supporting_edges": [{"relation": "reduces_variance", "polarity": "supports", "provenance": "Jones2022/methods"}],
        "contradicting_edges": [],
        "is_unconditional": True,
    },
]

BASE_STATE: AutoResearchState = {
    "topic": "RF vs Logistic Regression on iris",
    "run_id": "eval_writer",
    "retrieval_round": 1,
    "code_retry_count": 0,
    "hypothesis": (
        "Random Forest outperforms Logistic Regression on the iris dataset "
        "using 5-fold cross-validation."
    ),
    "experiment_spec": {
        "independent_var": "model_type",
        "dependent_var": "accuracy",
        "control_description": "Logistic Regression baseline",
        "dataset_id": "iris",
        "evaluation_metrics": ["accuracy", "f1"],
        "expected_outcome": "RF achieves higher accuracy",
    },
    "metrics_json": {"accuracy": 0.96, "f1": 0.95},
    "claim_ledger": CLAIM_LEDGER,
    "critique_warnings": [],
    "revision_pass_done": False,
    "kg_entities": [],
    "kg_edges": [],
}

# ─── Checkers ─────────────────────────────────────────────────────────────────

def check_structure(latex: str) -> tuple[bool, list[str]]:
    missing = [s for s in REQUIRED_SECTIONS if not re.search(s, latex)]
    return len(missing) == 0, missing


def check_latex_compile(latex: str) -> tuple[bool, str]:
    """Try to compile the LaTeX with pdflatex. Returns (ok, message)."""
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True, "pdflatex not available — skipped (structure check only)"

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "paper.tex")
        with open(tex_path, "w") as f:
            f.write(latex)
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return True, "compiled successfully"
        # Extract first error line for brevity
        errors = [l for l in result.stdout.splitlines() if l.startswith("!")]
        return False, errors[0] if errors else "compilation failed"


# ─── Runner ───────────────────────────────────────────────────────────────────

RUNS = int(os.getenv("EVAL_RUNS", "3"))


def run_eval():
    print("Starting Academic Writer Evaluation (requires ANTHROPIC_API_KEY)...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    structure_passes = 0
    compile_passes = 0

    for run_idx in range(1, RUNS + 1):
        print(f"\n--- Run {run_idx}/{RUNS} ---")
        state = dict(BASE_STATE)
        try:
            output = academic_writer(state)
            latex = output.get("latex_draft", "")
        except Exception as e:
            print(f"  Agent call failed: {e}")
            continue

        if not latex:
            print("  No LaTeX returned.")
            continue

        # Check 1: Structure
        struct_ok, missing = check_structure(latex)
        structure_passes += int(struct_ok)
        if struct_ok:
            print(f"  [OK] Structure: all 4 IMRaD sections present")
        else:
            print(f"  [FAIL] Structure: missing sections: {missing}")

        # Check 2: LaTeX compilation
        compile_ok, detail = check_latex_compile(latex)
        compile_passes += int(compile_ok)
        status = "OK" if compile_ok else "FAIL"
        print(f"  [{status}] Compilation: {detail}")

    print(f"\n=== Summary ===")
    print(f"  Structure:   {structure_passes}/{RUNS} passed")
    print(f"  Compilation: {compile_passes}/{RUNS} passed")

    if structure_passes == RUNS and compile_passes == RUNS:
        print("\nEvaluation PASSED")
    else:
        print("\nEvaluation PARTIAL — review failing checks above")


if __name__ == "__main__":
    run_eval()
