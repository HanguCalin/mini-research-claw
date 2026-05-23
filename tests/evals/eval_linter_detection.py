"""Eval: Linter Detection Rate.

Injects 10 known structural issues into LaTeX drafts and verifies
the deterministic_linter catches at least 9/10 of them.
No API key required — linter is fully deterministic.

Pass criteria: >= 9/10 issues detected.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.deterministic_linter import deterministic_linter

# ─── Gold-standard defects ────────────────────────────────────────────────────

GOOD_LATEX = r"""
\documentclass{article}
\begin{document}
\section{Introduction} Text \cite{Smith2020}.
\section{Methods} Methods here.
\section{Results} Results here.
\section{Conclusion} Conclusion here.
\end{document}
"""

GOOD_BIBTEX = "@article{Smith2020, author={Smith}, title={A Study}, year={2020}}"

DEFECT_CASES = [
    # 1. Missing Introduction
    {
        "name": "missing_introduction",
        "latex": r"\documentclass{article}\begin{document}\section{Methods}M.\section{Results}R.\section{Conclusion}C.\end{document}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "imrad_completeness",
    },
    # 2. Missing Methods
    {
        "name": "missing_methods",
        "latex": r"\documentclass{article}\begin{document}\section{Introduction}I.\section{Results}R.\section{Conclusion}C.\end{document}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "imrad_completeness",
    },
    # 3. Missing Results
    {
        "name": "missing_results",
        "latex": r"\documentclass{article}\begin{document}\section{Introduction}I.\section{Methods}M.\section{Conclusion}C.\end{document}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "imrad_completeness",
    },
    # 4. Missing Conclusion
    {
        "name": "missing_conclusion",
        "latex": r"\documentclass{article}\begin{document}\section{Introduction}I.\section{Methods}M.\section{Results}R.\end{document}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "imrad_completeness",
    },
    # 5. Orphaned citation (cite key not in bibtex)
    {
        "name": "orphaned_citation",
        "latex": GOOD_LATEX + r" Also see \cite{Missing2099}.",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "citation_integrity",
    },
    # 6. Weak claim in ledger included in draft
    {
        "name": "weak_claim_included",
        "latex": GOOD_LATEX + " RF wins always.",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [
            {
                "claim_id": "c1",
                "claim_text": "RF wins always.",
                "supporting_kg_edges": [],
                "contradicting_kg_edges": [],
                "evidence_strength": "weak",
            }
        ],
        "expected_check": "claim_ledger_compliance",
    },
    # 7. Unsupported claim in ledger included in draft
    {
        "name": "unsupported_claim_included",
        "latex": GOOD_LATEX + " Magic results observed.",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [
            {
                "claim_id": "c2",
                "claim_text": "Magic results observed.",
                "supporting_kg_edges": [],
                "contradicting_kg_edges": [],
                "evidence_strength": "unsupported",
            }
        ],
        "expected_check": "claim_ledger_compliance",
    },
    # 8. Raw arXiv ID in prose
    {
        "name": "raw_arxiv_id",
        "latex": GOOD_LATEX + r" As shown in arXiv:2301.00001, this works.",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "raw_arxiv_ids",
    },
    # 9. Figure without label
    {
        "name": "figure_without_label",
        "latex": GOOD_LATEX + r"\begin{figure}\caption{Result}\end{figure}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "figure_table_labels",
    },
    # 10. Table without caption
    {
        "name": "table_without_caption",
        "latex": GOOD_LATEX + r"\begin{table}\label{tab:1}Data here.\end{table}",
        "bibtex": GOOD_BIBTEX,
        "claim_ledger": [],
        "expected_check": "figure_table_labels",
    },
]

# ─── Eval Logic ───────────────────────────────────────────────────────────────

def run_eval():
    print("Starting Linter Detection Rate Evaluation (no API key required)...")
    detected = 0
    results = []

    for case in DEFECT_CASES:
        state = {
            "latex_draft": case["latex"],
            "bibtex_source": case["bibtex"],
            "claim_ledger": case["claim_ledger"],
            "critique_warnings": [],
        }
        result = deterministic_linter(state)
        warnings = result.get("critique_warnings", [])
        checks_fired = {w["check"] for w in warnings}
        passed = case["expected_check"] in checks_fired
        detected += int(passed)
        results.append((case["name"], passed, checks_fired))
        status = "DETECTED" if passed else "MISSED"
        print(f"  [{status}] {case['name']} (expected check: {case['expected_check']})")

    print(f"\n--- Results ---")
    print(f"Detected: {detected}/{len(DEFECT_CASES)}")

    if detected >= 9:
        print("\nEvaluation PASSED (>= 9/10 issues detected)")
    else:
        print(f"\nEvaluation FAILED ({detected}/10 — threshold is 9)")


if __name__ == "__main__":
    run_eval()
