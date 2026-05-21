"""Node 7: Critique Panel (3 Heterogeneous Agents + Structured Debate).

Type: AI (3 agents with different models/personas).
Agent A: Fact-Checker (Sonnet) — verifies claims against KG.
Agent B: Methodologist (Haiku) — evaluates experimental rigor.
Agent C: Formatter (Haiku) — assesses writing quality.

Debate protocol: independent critique → cross-challenge →
defend-or-retract → resolution (only unretracted critiques survive).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from backend.config import MODELS
from backend.state import AutoResearchState, DebateEntry
from backend.utils.llm_utils import extract_json, extract_text

logger = logging.getLogger(__name__)

# ─── Agent system prompts ────────────────────────────────────────────────────

FACT_CHECKER_PROMPT = """\
You are a FACT-CHECKER reviewing an ML research paper.

You have access to the Knowledge Graph and Claim Ledger as structured data.
You MUST use JSON path traversals to query kg_entities/kg_edges/claim_ledger —
do NOT rely on parametric memory. Cite specific entity IDs and edge relations.

Flag these issues:
- "ungrounded": claims not backed by KG edges
- "contradiction_suppressed": contradicting KG edges exist but aren't acknowledged

Return JSON array of critiques:
[{"critique": "...", "severity": "error|warning", "evidence": "..."}]
"""

METHODOLOGIST_PROMPT = """\
You are a METHODOLOGIST reviewing an ML research paper.

Check if metrics.json results logically support the paper's conclusions.
Verify ExperimentSpec compliance. Flag:
- Unsupported conclusions
- Missing error bars or confidence intervals
- Statistical issues (no significance tests, wrong metrics)
- Mismatches between claimed and measured results

Return JSON array of critiques:
[{"critique": "...", "severity": "error|warning", "evidence": "..."}]
"""

FORMATTER_PROMPT = """\
You are a WRITING QUALITY reviewer for an ML research paper.

Focus on SUBJECTIVE quality only (structural checks are handled by the linter):
- AI-slop writing style (vague superlatives, filler phrases)
- Verbosity and redundancy
- Argumentation flow and logical coherence
- Clarity of technical exposition

Return JSON array of critiques:
[{"critique": "...", "severity": "warning", "evidence": "..."}]
"""

CHALLENGE_PROMPT = """\
You are reviewing another agent's critiques of a research paper.
For each critique you DISAGREE with, issue a formal challenge explaining
why the critique is incorrect, excessive, or unfounded.

Critiques to review:
{critiques}

The paper context:
{paper_context}

Return JSON array (empty if you agree with everything):
[{{"target_critique_index": 0, "challenge": "..."}}]
"""

DEFEND_PROMPT = """\
Your critique has been challenged. You MUST either:
1. DEFEND with specific evidence from the paper/KG, or
2. RETRACT if the challenge is valid.

Your original critique: {critique}
Challenge: {challenge}

Return JSON: {{"action": "defend|retract", "response": "..."}}
"""


def critique_panel(state: AutoResearchState) -> dict[str, Any]:
    """Run the 3-agent critique panel with structured debate protocol."""
    client = anthropic.Anthropic()

    latex_draft = state.get("latex_draft", "")
    bibtex = state.get("bibtex_source", "")
    metrics = state.get("metrics_json", {})
    claim_ledger = state.get("claim_ledger", [])
    kg_entities = state.get("kg_entities", [])
    kg_edges = state.get("kg_edges", [])

    paper_context = (
        f"LaTeX draft (first 3000 chars):\n{latex_draft[:3000]}\n\n"
        f"Metrics: {json.dumps(metrics, indent=2)}\n\n"
        f"Claim ledger: {json.dumps(claim_ledger[:10], indent=2)}"
    )

    # ── Phase 1: Independent critique ────────────────────────────────────
    agents = [
        ("fact_checker", MODELS.critique_fact_checker, FACT_CHECKER_PROMPT, {
            "kg_entities": json.dumps(kg_entities[:30], indent=2),
            "kg_edges": json.dumps(kg_edges[:50], indent=2),
            "claim_ledger": json.dumps(claim_ledger, indent=2),
        }),
        ("methodologist", MODELS.critique_methodologist, METHODOLOGIST_PROMPT, {
            "metrics": json.dumps(metrics, indent=2),
        }),
        ("formatter", MODELS.critique_formatter, FORMATTER_PROMPT, {}),
    ]

    all_critiques: dict[str, list[dict[str, Any]]] = {}

    for role, model, system_prompt, extra_context in agents:
        user_msg = paper_context
        for key, val in extra_context.items():
            user_msg += f"\n\n{key}:\n{val}"

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            critiques = extract_json(extract_text(response))
            all_critiques[role] = critiques if isinstance(critiques, list) else []
        except (ValueError, anthropic.APIError) as exc:
            logger.warning("Critique from %s failed: %s", role, exc)
            all_critiques[role] = []

    # ── Phase 2: Cross-challenge ─────────────────────────────────────────
    roles = list(all_critiques.keys())
    challenges: list[dict[str, Any]] = []

    for challenger_role in roles:
        other_critiques: list[dict[str, Any]] = []
        for r in roles:
            if r != challenger_role:
                for i, c in enumerate(all_critiques[r]):
                    other_critiques.append({"index": i, "from": r, **c})

        if not other_critiques:
            continue

        prompt = CHALLENGE_PROMPT.format(
            critiques=json.dumps(other_critiques, indent=2),
            paper_context=paper_context[:2000],
        )

        try:
            response = client.messages.create(
                model=MODELS.critique_methodologist,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_challenges = extract_json(extract_text(response))
            for ch in raw_challenges:
                ch["challenger_role"] = challenger_role
            challenges.extend(raw_challenges)
        except (ValueError, anthropic.APIError):
            pass

    # ── Phase 3: Defend-or-retract ───────────────────────────────────────
    debate_log: list[DebateEntry] = []
    retracted_indices: set[tuple[str, int]] = set()

    for challenge in challenges:
        target_idx = challenge.get("target_critique_index", 0)
        challenger = challenge.get("challenger_role", "unknown")
        challenge_text = challenge.get("challenge", "")

        target_role = None
        for r in roles:
            if r != challenger and target_idx < len(all_critiques.get(r, [])):
                target_role = r
                break

        if not target_role:
            continue

        original_critique = all_critiques[target_role][target_idx]

        prompt = DEFEND_PROMPT.format(
            critique=json.dumps(original_critique),
            challenge=challenge_text,
        )

        try:
            response = client.messages.create(
                model=MODELS.critique_methodologist,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            result = extract_json(extract_text(response))
            action = result.get("action", "defend")
            response_text = result.get("response", "")

            resolved = action == "retract"
            if resolved:
                retracted_indices.add((target_role, target_idx))

            debate_log.append(DebateEntry(
                round=1,
                challenger_role=challenger,
                target_critique_index=target_idx,
                challenge=challenge_text,
                response=response_text,
                resolved=resolved,
            ))
        except (ValueError, anthropic.APIError):
            pass

    # ── Phase 4: Resolution ──────────────────────────────────────────────
    surviving: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []

    for role, critiques in all_critiques.items():
        for i, critique in enumerate(critiques):
            critique["source"] = role
            all_warnings.append(critique)
            if (role, i) not in retracted_indices:
                surviving.append(critique)

    logger.info(
        "Critique panel: %d total critiques, %d retracted, %d surviving",
        len(all_warnings), len(retracted_indices), len(surviving),
    )

    return {
        "critique_warnings": state.get("critique_warnings", []) + all_warnings,
        "debate_log": debate_log,
        "surviving_critiques": surviving,
    }
