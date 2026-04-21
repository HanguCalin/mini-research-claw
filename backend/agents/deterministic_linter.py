"""Node 6b: Deterministic Linter.

Type: Non-AI (regex + LaTeX parsing).
Runs 6 objective checks on the LaTeX draft. Linter warnings bypass the
debate protocol — they are non-debatable.
"""

from __future__ import annotations

import re
from typing import Any

from backend.state import AutoResearchState


def deterministic_linter(state: AutoResearchState) -> dict[str, Any]:
    """Run all 6 deterministic checks on the draft."""
    latex = state.get("latex_draft", "")
    bibtex = state.get("bibtex_source", "")
    claim_ledger = state.get("claim_ledger", [])
    existing_warnings: list[dict[str, Any]] = list(state.get("critique_warnings", []))

    warnings = _check_imrad(latex)
    warnings += _check_citation_integrity(latex, bibtex)
    warnings += _check_claim_ledger_compliance(latex, claim_ledger)
    warnings += _check_neurips_checklist(latex)
    warnings += _check_figure_table_labels(latex)
    warnings += _check_raw_arxiv_ids(latex)

    for w in warnings:
        w["source"] = "linter"

    existing_warnings.extend(warnings)
    return {"critique_warnings": existing_warnings}


# ─── Check 1: IMRaD completeness ────────────────────────────────────────────

REQUIRED_SECTIONS = [
    r"\\section\{Introduction\}",
    r"\\section\{Methods\}",
    r"\\section\{Results\}",
    r"\\section\{Conclusion\}",
]


def _check_imrad(latex: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for pattern in REQUIRED_SECTIONS:
        if not re.search(pattern, latex):
            section_name = re.search(r"\{(.+)\}", pattern)
            name = section_name.group(1) if section_name else pattern
            warnings.append({
                "check": "imrad_completeness",
                "severity": "error",
                "message": f"Missing required section: \\section{{{name}}}",
            })
    return warnings


# ─── Check 2: Citation integrity ────────────────────────────────────────────

def _check_citation_integrity(latex: str, bibtex: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    cite_keys = set(re.findall(r"\\cite\{([^}]+)\}", latex))
    expanded_keys: set[str] = set()
    for key_group in cite_keys:
        for k in key_group.split(","):
            expanded_keys.add(k.strip())

    bib_keys = set(re.findall(r"@\w+\{(\w+),", bibtex))

    orphaned = expanded_keys - bib_keys
    for key in sorted(orphaned):
        warnings.append({
            "check": "citation_integrity",
            "severity": "error",
            "message": f"Orphaned citation: \\cite{{{key}}} has no BibTeX entry",
        })
    return warnings


# ─── Check 3: Claim-ledger compliance ───────────────────────────────────────

def _check_claim_ledger_compliance(
    latex: str,
    claim_ledger: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    latex_lower = latex.lower()

    for claim in claim_ledger:
        strength = claim.get("evidence_strength", "")
        if strength not in ("weak", "unsupported"):
            continue

        claim_text = claim.get("claim_text", "")
        words = claim_text.lower().split()
        key_phrase = " ".join(words[:6]) if len(words) >= 6 else claim_text.lower()

        if key_phrase in latex_lower:
            warnings.append({
                "check": "claim_ledger_compliance",
                "severity": "warning",
                "message": (
                    f"Draft includes {strength}-evidence claim: "
                    f'"{claim_text[:80]}..."'
                ),
            })
    return warnings


# ─── Check 4: NeurIPS checklist ─────────────────────────────────────────────

def _check_neurips_checklist(latex: str) -> list[dict[str, Any]]:
    if "reproducibility" not in latex.lower() and "checklist" not in latex.lower():
        return [{
            "check": "neurips_checklist",
            "severity": "warning",
            "message": "Missing NeurIPS reproducibility checklist section",
        }]
    return []


# ─── Check 5: Figure/table labeling ─────────────────────────────────────────

def _check_figure_table_labels(latex: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    for env in ("figure", "table"):
        pattern = re.compile(
            rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
            re.DOTALL,
        )
        for match in pattern.finditer(latex):
            block = match.group(1)
            if "\\label{" not in block:
                warnings.append({
                    "check": "figure_table_labels",
                    "severity": "warning",
                    "message": f"{env} environment missing \\label{{}}",
                })
            if "\\caption{" not in block:
                warnings.append({
                    "check": "figure_table_labels",
                    "severity": "warning",
                    "message": f"{env} environment missing \\caption{{}}",
                })
    return warnings


# ─── Check 6: No raw arXiv IDs ──────────────────────────────────────────────

def _check_raw_arxiv_ids(latex: str) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []

    bibtex_blocks = set()
    for m in re.finditer(r"@\w+\{.*?\}", latex, re.DOTALL):
        bibtex_blocks.add((m.start(), m.end()))

    for m in re.finditer(r"arXiv:\d{4}\.\d{4,5}", latex):
        in_bibtex = any(start <= m.start() <= end for start, end in bibtex_blocks)
        if not in_bibtex:
            warnings.append({
                "check": "raw_arxiv_ids",
                "severity": "warning",
                "message": f"Raw arXiv ID in prose: {m.group()} — use \\cite{{}} instead",
            })
    return warnings
