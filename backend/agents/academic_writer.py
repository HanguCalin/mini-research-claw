"""Node 6: Academic Writer.

Type: AI (Claude Sonnet).
First pass: generates IMRaD LaTeX draft + BibTeX from claim ledger.
Revision pass: addresses critique warnings, appends confidence score
and NeurIPS reproducibility checklist. Exactly one revision pass.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from backend.config import MODELS
from backend.state import AutoResearchState

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """\
You are an academic paper writer producing LaTeX for a machine learning paper.

STRUCTURE: IMRaD format with these exact sections:
\\section{Introduction}
\\section{Methods}
\\section{Results}
\\section{Conclusion}

RULES:
1. ONLY include claims rated "strong" or "moderate" in the claim ledger.
   Do NOT include "weak" or "unsupported" claims.
2. For claims with contradicting evidence, ACKNOWLEDGE the contradiction
   explicitly in the text.
3. Use \\cite{AuthorYear} citations — NO raw arXiv IDs in prose.
4. Generate a companion references.bib with proper BibTeX entries.
5. Include \\label{} and \\caption{} for every figure and table.

Return your output as JSON:
{
  "latex_draft": "\\\\documentclass{article}...",
  "bibtex_source": "@article{...}..."
}
"""

REVISION_SYSTEM_PROMPT = """\
You are revising an academic ML paper based on critique feedback.

RULES:
1. Address EVERY critique in the feedback list.
2. Do not remove valid content — only fix, clarify, or strengthen.
3. After revision, append a Confidence Score section (self-assessed 1-10)
   and the NeurIPS reproducibility checklist.
4. This is the FINAL revision — make it count.

Return your output as JSON:
{
  "latex_draft": "\\\\documentclass{article}...",
  "confidence_score": 7.5
}
"""


def academic_writer(state: AutoResearchState) -> dict[str, Any]:
    """Generate or revise the LaTeX draft."""
    revision_pass_done = state.get("revision_pass_done", False)

    if revision_pass_done:
        return _revision_pass(state)
    return _first_pass(state)


def _first_pass(state: AutoResearchState) -> dict[str, Any]:
    """Generate initial IMRaD LaTeX draft + BibTeX."""
    client = anthropic.Anthropic()

    claim_ledger = state.get("claim_ledger", [])
    spec = state.get("experiment_spec", {})
    metrics = state.get("metrics_json", {})
    delta = state.get("incremental_delta", "")
    hypothesis = state.get("hypothesis", "")

    strong_moderate = [
        c for c in claim_ledger
        if c.get("evidence_strength") in ("strong", "moderate")
    ]

    user_prompt = (
        f"Hypothesis: {hypothesis}\n\n"
        f"Incremental delta: {delta}\n\n"
        f"Experiment spec: {json.dumps(spec, indent=2)}\n\n"
        f"Experiment metrics: {json.dumps(metrics, indent=2)}\n\n"
        f"Claim ledger (strong/moderate only):\n"
        f"{json.dumps(strong_moderate, indent=2)}\n\n"
        "Write the full LaTeX paper and BibTeX references."
    )

    response = client.messages.create(
        model=MODELS.academic_writer,
        max_tokens=16384,
        system=WRITER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = json.loads(response.content[0].text)

    return {
        "latex_draft": raw["latex_draft"],
        "bibtex_source": raw.get("bibtex_source", ""),
        "revision_pass_done": False,
    }


def _revision_pass(state: AutoResearchState) -> dict[str, Any]:
    """Revise the draft based on aggregated critique warnings."""
    client = anthropic.Anthropic()

    latex_draft = state.get("latex_draft", "")
    critiques = state.get("critique_warnings", [])
    surviving = state.get("surviving_critiques", [])

    all_feedback = critiques + surviving

    user_prompt = (
        f"Current LaTeX draft:\n{latex_draft}\n\n"
        f"Critique feedback to address:\n{json.dumps(all_feedback, indent=2)}\n\n"
        "Revise the paper addressing every critique."
    )

    response = client.messages.create(
        model=MODELS.academic_writer,
        max_tokens=16384,
        system=REVISION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = json.loads(response.content[0].text)

    return {
        "latex_draft": raw["latex_draft"],
        "confidence_score": float(raw.get("confidence_score", 5.0)),
        "revision_pass_done": True,
    }
