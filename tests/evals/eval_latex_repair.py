"""Eval: LaTeX Repair Loop fix rate — §7.3.

Injects 10 common LaTeX errors and verifies the repair loop fixes them.
Requires pdflatex installed AND ANTHROPIC_API_KEY (repair uses Claude Haiku).

Pass criteria: >= 7/10 errors fixed within max_latex_repair_attempts.

Note: if pdflatex is not installed this eval skips gracefully.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.latex_compiler import latex_compiler
from backend.state import AutoResearchState

# ─── Minimal valid LaTeX template ────────────────────────────────────────────

_VALID_DOC = r"""\documentclass{article}
\begin{document}
\section{Introduction}
This paper evaluates Random Forest vs Logistic Regression.
\section{Methods}
We use 5-fold cross-validation on the iris dataset.
\section{Results}
RF achieves 0.96 accuracy.
\section{Conclusion}
RF outperforms LR on small datasets.
\end{document}
"""

# ─── Error injections ────────────────────────────────────────────────────────
# Each tuple: (description, broken_latex)
# The repair loop must fix each one within max_latex_repair_attempts.

def _inject(original: str, old: str, new: str) -> str:
    return original.replace(old, new, 1)


ERROR_CASES = [
    (
        "Unclosed \\begin{itemize}",
        _inject(_VALID_DOC,
                r"\section{Results}" + "\nRF achieves 0.96 accuracy.",
                r"\section{Results}" + "\n\\begin{itemize}\n\\item RF achieves 0.96 accuracy."),
    ),
    (
        "Missing \\end{document}",
        _VALID_DOC.replace(r"\end{document}", ""),
    ),
    (
        "Undefined control sequence \\badcommand",
        _inject(_VALID_DOC, "This paper evaluates", r"\badcommand This paper evaluates"),
    ),
    (
        "Unmatched $ in math mode",
        _inject(_VALID_DOC, "0.96 accuracy", "$0.96 accuracy"),
    ),
    (
        "Missing \\begin{document}",
        _VALID_DOC.replace(r"\begin{document}", ""),
    ),
    (
        "Double \\end{document}",
        _VALID_DOC.replace(r"\end{document}", r"\end{document}" + "\n" + r"\end{document}"),
    ),
    (
        "Misspelled environment \\begin{eqaution}",
        _inject(_VALID_DOC, "RF achieves 0.96 accuracy.",
                r"\begin{eqaution} acc = 0.96 \end{eqaution}"),
    ),
    (
        "Unbalanced braces in \\section",
        _inject(_VALID_DOC, r"\section{Introduction}", r"\section{Introduction"),
    ),
    (
        "Invalid \\cite without biblio",
        _inject(_VALID_DOC, "This paper evaluates", r"As shown in \cite{Fake2099}, this paper evaluates"),
    ),
    (
        "\\usepackage after \\begin{document}",
        _inject(_VALID_DOC, r"\begin{document}", r"\begin{document}" + "\n" + r"\usepackage{amsmath}"),
    ),
]

# ─── Runner ───────────────────────────────────────────────────────────────────

def _pdflatex_available() -> bool:
    import subprocess
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_eval():
    print("Starting LaTeX Repair Eval (requires pdflatex + ANTHROPIC_API_KEY)...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    if not _pdflatex_available():
        print("SKIP: pdflatex not found — install TeX Live or MiKTeX to run this eval.")
        return

    fixed = 0
    total = len(ERROR_CASES)

    for idx, (desc, broken_latex) in enumerate(ERROR_CASES, 1):
        print(f"\n--- Case {idx}/{total}: {desc} ---")
        state: AutoResearchState = {
            "topic": "eval",
            "run_id": f"eval_latex_repair_{idx}",
            "retrieval_round": 1,
            "code_retry_count": 0,
            "latex_draft": broken_latex,
            "bibtex_source": "",
        }
        try:
            result = latex_compiler(state)
            if result.get("pipeline_status") == "success":
                fixed += 1
                attempts = result.get("latex_repair_attempts", 0)
                print(f"  [OK] Fixed in {attempts} repair attempt(s)")
            else:
                print(f"  [FAIL] Could not fix within repair budget")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n=== Summary ===")
    print(f"  Fixed: {fixed}/{total}")

    if fixed >= 7:
        print(f"\nEvaluation PASSED (>= 7/10 fixed)")
    else:
        print(f"\nEvaluation FAILED ({fixed}/10 — threshold is 7)")


if __name__ == "__main__":
    run_eval()
