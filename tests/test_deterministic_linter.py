"""Unit tests for backend/agents/deterministic_linter.py

Tests all 6 deterministic checks:
1. IMRaD completeness
2. Citation integrity (orphaned \\cite keys)
3. Claim-ledger compliance (weak/unsupported evidence in draft)
4. NeurIPS reproducibility checklist
5. Figure/table label and caption presence
6. No raw arXiv IDs in prose
"""

import pytest

from backend.agents.deterministic_linter import (
    _check_citation_integrity,
    _check_claim_ledger_compliance,
    _check_figure_table_labels,
    _check_imrad,
    _check_neurips_checklist,
    _check_raw_arxiv_ids,
    deterministic_linter,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

FULL_LATEX = r"""
\section{Introduction}
Some intro text \cite{ref1}.
\section{Methods}
Methods here.
\section{Results}
Results here.
\section{Conclusion}
Conclusion here.
reproducibility checklist included.
"""

FULL_BIBTEX = r"@article{ref1, title={A paper}, author={Author}}"


# ─── Check 1: IMRaD completeness ─────────────────────────────────────────────


def test_imrad_no_warnings_on_complete_draft():
    assert _check_imrad(FULL_LATEX) == []


def test_imrad_flags_missing_introduction():
    latex = r"\section{Methods}\section{Results}\section{Conclusion}"
    warnings = _check_imrad(latex)
    checks = [w["check"] for w in warnings]
    assert "imrad_completeness" in checks
    assert any("Introduction" in w["message"] for w in warnings)


def test_imrad_flags_missing_methods():
    latex = r"\section{Introduction}\section{Results}\section{Conclusion}"
    warnings = _check_imrad(latex)
    assert any("Methods" in w["message"] for w in warnings)


def test_imrad_flags_missing_results():
    latex = r"\section{Introduction}\section{Methods}\section{Conclusion}"
    warnings = _check_imrad(latex)
    assert any("Results" in w["message"] for w in warnings)


def test_imrad_flags_missing_conclusion():
    latex = r"\section{Introduction}\section{Methods}\section{Results}"
    warnings = _check_imrad(latex)
    assert any("Conclusion" in w["message"] for w in warnings)


def test_imrad_flags_all_missing_on_empty():
    warnings = _check_imrad("")
    assert len(warnings) == 4


def test_imrad_warnings_have_error_severity():
    warnings = _check_imrad("")
    assert all(w["severity"] == "error" for w in warnings)


# ─── Check 2: Citation integrity ─────────────────────────────────────────────


def test_citation_no_warnings_when_all_defined():
    latex = r"\cite{ref1}"
    bibtex = r"@article{ref1, title={X}}"
    assert _check_citation_integrity(latex, bibtex) == []


def test_citation_flags_orphaned_key():
    latex = r"\cite{missing_ref}"
    bibtex = r"@article{ref1, title={X}}"
    warnings = _check_citation_integrity(latex, bibtex)
    assert len(warnings) == 1
    assert "missing_ref" in warnings[0]["message"]


def test_citation_flags_multiple_orphaned_keys():
    latex = r"\cite{a, b, c}"
    bibtex = r"@article{a, title={X}}"
    warnings = _check_citation_integrity(latex, bibtex)
    orphaned = {w["message"] for w in warnings}
    assert any("b" in m for m in orphaned)
    assert any("c" in m for m in orphaned)


def test_citation_no_cites_no_warnings():
    assert _check_citation_integrity("No citations here.", "") == []


def test_citation_warnings_have_error_severity():
    warnings = _check_citation_integrity(r"\cite{x}", "")
    assert all(w["severity"] == "error" for w in warnings)


# ─── Check 3: Claim-ledger compliance ────────────────────────────────────────


def test_claim_ledger_no_warnings_on_strong_evidence():
    claim = {
        "claim_id": "c1",
        "claim_text": "Random Forest outperforms XGBoost on tabular data",
        "evidence_strength": "strong",
    }
    warnings = _check_claim_ledger_compliance(
        "Random Forest outperforms XGBoost on tabular data", [claim]
    )
    assert warnings == []


def test_claim_ledger_flags_weak_evidence_in_draft():
    claim = {
        "claim_id": "c1",
        "claim_text": "Our method achieves state of the art performance",
        "evidence_strength": "weak",
    }
    latex = "our method achieves state of the art performance on all benchmarks"
    warnings = _check_claim_ledger_compliance(latex, [claim])
    assert len(warnings) == 1
    assert warnings[0]["check"] == "claim_ledger_compliance"


def test_claim_ledger_flags_unsupported_evidence():
    claim = {
        "claim_id": "c2",
        "claim_text": "This approach is universally superior",
        "evidence_strength": "unsupported",
    }
    latex = "this approach is universally superior in all cases"
    warnings = _check_claim_ledger_compliance(latex, [claim])
    assert len(warnings) == 1


def test_claim_ledger_no_warnings_on_empty_ledger():
    assert _check_claim_ledger_compliance("some draft text", []) == []


def test_claim_ledger_no_flag_when_claim_not_in_draft():
    claim = {
        "claim_id": "c1",
        "claim_text": "Some obscure claim not in the draft at all",
        "evidence_strength": "weak",
    }
    assert _check_claim_ledger_compliance("totally different content", [claim]) == []


# ─── Check 4: NeurIPS checklist ───────────────────────────────────────────────


def test_neurips_no_warning_when_reproducibility_present():
    assert _check_neurips_checklist("reproducibility section here") == []


def test_neurips_no_warning_when_checklist_present():
    assert _check_neurips_checklist("see the checklist below") == []


def test_neurips_flags_missing_checklist():
    warnings = _check_neurips_checklist("no mention of anything relevant")
    assert len(warnings) == 1
    assert warnings[0]["check"] == "neurips_checklist"


def test_neurips_warning_is_warning_severity():
    warnings = _check_neurips_checklist("nothing relevant in this draft")
    assert warnings[0]["severity"] == "warning"


# ─── Check 5: Figure/table labels ─────────────────────────────────────────────


def test_figure_no_warning_with_label_and_caption():
    latex = r"\begin{figure}\label{fig1}\caption{My figure}\end{figure}"
    assert _check_figure_table_labels(latex) == []


def test_figure_flags_missing_label():
    latex = r"\begin{figure}\caption{My figure}\end{figure}"
    warnings = _check_figure_table_labels(latex)
    assert any("missing \\label" in w["message"] for w in warnings)


def test_figure_flags_missing_caption():
    latex = r"\begin{figure}\label{fig1}\end{figure}"
    warnings = _check_figure_table_labels(latex)
    assert any("missing \\caption" in w["message"] for w in warnings)


def test_table_flags_missing_label_and_caption():
    latex = r"\begin{table}some content\end{table}"
    warnings = _check_figure_table_labels(latex)
    checks = [w["message"] for w in warnings]
    assert any("label" in m for m in checks)
    assert any("caption" in m for m in checks)


def test_no_figures_no_warnings():
    assert _check_figure_table_labels("No environments here.") == []


# ─── Check 6: Raw arXiv IDs ──────────────────────────────────────────────────


def test_raw_arxiv_flags_id_in_prose():
    latex = "As shown in arXiv:2401.12345, the method works."
    warnings = _check_raw_arxiv_ids(latex)
    assert len(warnings) == 1
    assert "arXiv:2401.12345" in warnings[0]["message"]


def test_raw_arxiv_no_warning_when_no_ids():
    assert _check_raw_arxiv_ids("No arXiv references here.") == []


def test_raw_arxiv_flags_multiple_ids():
    latex = "See arXiv:2401.12345 and also arXiv:2402.67890 for details."
    warnings = _check_raw_arxiv_ids(latex)
    assert len(warnings) == 2


# ─── deterministic_linter node (full integration) ────────────────────────────


def test_linter_returns_critique_warnings_key():
    state = {"latex_draft": FULL_LATEX, "bibtex_source": FULL_BIBTEX, "claim_ledger": []}
    result = deterministic_linter(state)
    assert "critique_warnings" in result


def test_linter_appends_source_field_to_all_warnings():
    state = {"latex_draft": "", "bibtex_source": "", "claim_ledger": []}
    result = deterministic_linter(state)
    for w in result["critique_warnings"]:
        assert w["source"] == "linter"


def test_linter_preserves_existing_warnings():
    existing = [{"check": "prior_check", "severity": "info", "message": "existing"}]
    state = {
        "latex_draft": FULL_LATEX,
        "bibtex_source": FULL_BIBTEX,
        "claim_ledger": [],
        "critique_warnings": existing,
    }
    result = deterministic_linter(state)
    assert any(w["check"] == "prior_check" for w in result["critique_warnings"])


def test_linter_clean_draft_has_minimal_warnings():
    state = {
        "latex_draft": FULL_LATEX + "\nchecklist and reproducibility",
        "bibtex_source": FULL_BIBTEX,
        "claim_ledger": [],
    }
    result = deterministic_linter(state)
    errors = [w for w in result["critique_warnings"] if w["severity"] == "error"]
    assert len(errors) == 0
