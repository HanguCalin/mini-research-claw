"""Eval §7.3 — Critique Panel: Revision Quality.

Methodology:
  1. Feed a deliberately weak pre-revision draft to the critique panel.
  2. Pass the surviving critiques to the academic_writer (revision pass).
  3. Use Claude (judge model) to score the improvement on a 1–5 scale.
  4. Pass criterion: average improvement score >= 3.0/5.

Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import anthropic

from backend.agents.academic_writer import academic_writer
from backend.agents.critique_panel import critique_panel

# ─── Deliberately weak pre-revision draft ────────────────────────────────────

PRE_REVISION_DRAFT = r"""
\documentclass{article}
\begin{document}

\section{Introduction}
We studied some classifiers on data.

\section{Methods}
We used some methods. No details are provided on cross-validation,
hyperparameter selection, or train/test split strategy.

\section{Results}
Results were good. Accuracy was high. The method worked well.

\section{Conclusion}
Our approach is effective. Future work will improve things.

\end{document}
"""

KG_ENTITIES = [
    {"id": "e1", "canonical_name": "Random Forest",
     "entity_type": "model", "aliases": ["RF"], "attributes": {}},
    {"id": "e2", "canonical_name": "XGBoost",
     "entity_type": "model", "aliases": ["XGB"], "attributes": {}},
    {"id": "e3", "canonical_name": "Iris",
     "entity_type": "dataset", "aliases": ["iris dataset"], "attributes": {}},
]

KG_EDGES = [
    {
        "source_id": "e1", "target_id": "e2",
        "relation": "outperforms", "polarity": "supports",
        "context_condition": "",
        "confidence": 0.88, "provenance": "Smith2024/results",
    },
    {
        "source_id": "e2", "target_id": "e1",
        "relation": "outperforms", "polarity": "supports",
        "context_condition": "when hyperparameters are tuned via Bayesian optimization",
        "confidence": 0.82, "provenance": "Jones2023/results",
    },
]

CLAIM_LEDGER = [
    {
        "claim_id": "c1",
        "claim_text": "Random Forest outperforms XGBoost on the Iris dataset.",
        "evidence_strength": "moderate",
        "supporting_kg_edges": ["e1→e2"],
        "contradicting_kg_edges": ["e2→e1"],
    },
    {
        "claim_id": "c2",
        "claim_text": "5-fold cross-validation reduces variance in accuracy estimates.",
        "evidence_strength": "strong",
        "supporting_kg_edges": [],
        "contradicting_kg_edges": [],
    },
]

BASE_STATE = {
    "topic": "Random Forest vs XGBoost on tabular classification",
    "run_id": "eval_critique_revision",
    "retrieval_round": 1,
    "code_retry_count": 0,
    "hypothesis": (
        "Random Forest outperforms XGBoost on the Iris dataset "
        "using 5-fold cross-validation, with accuracy >= 0.92."
    ),
    "incremental_delta": (
        "We control for Bayesian hyperparameter tuning, resolving the "
        "contested outperforms relationship between RF and XGBoost."
    ),
    "experiment_spec": {
        "independent_var": "classifier_type",
        "dependent_var": "accuracy",
        "control_description": "XGBoost with default hyperparameters",
        "dataset_id": "iris",
        "evaluation_metrics": ["accuracy", "f1_macro"],
        "expected_outcome": "RF achieves accuracy >= 0.92 vs XGBoost <= 0.89",
    },
    "metrics_json": {"accuracy": 0.947, "f1_macro": 0.946},
    "claim_ledger": CLAIM_LEDGER,
    "kg_entities": KG_ENTITIES,
    "kg_edges": KG_EDGES,
    "bibtex_source": (
        "@article{Smith2024, author={Smith, A.}, title={RF vs XGBoost}, "
        "year={2024}, journal={JMLR}}\n"
        "@article{Jones2023, author={Jones, B.}, title={Boosting Comparison}, "
        "year={2023}, journal={NeurIPS}}"
    ),
    "critique_warnings": [],
    "surviving_critiques": [],
    "revision_pass_done": False,
    "latex_draft": PRE_REVISION_DRAFT,
}

# ─── Claude-as-judge configuration ───────────────────────────────────────────

JUDGE_SYSTEM = """\
You are an expert academic paper reviewer evaluating the quality improvement
between a pre-revision and post-revision version of a research paper.

Score the improvement on a 1–5 scale:
  1 = No improvement, or the revised version is worse
  2 = Minor surface-level improvements (wording, spacing) but no substance
  3 = Noticeable improvements: clearer methodology, better structure,
      or more precise claims compared to the pre-revision draft
  4 = Significant improvements: major methodological clarifications,
      acknowledged contradictions, stronger evidence attribution
  5 = Exceptional improvement: the post-revision draft is substantially
      more rigorous, complete, and scientifically sound

Respond with ONLY a single integer (1, 2, 3, 4, or 5).
Do NOT include any explanation, reasoning, or other text.\
"""

JUDGE_USER_TEMPLATE = """\
Pre-revision draft (first {pre_chars} chars):
{pre_revision}

Post-revision draft (first {post_chars} chars):
{post_revision}

Score the improvement (1-5):\
"""

PASS_THRESHOLD = 3.0
RUNS = int(os.getenv("EVAL_RUNS", "3"))
_MAX_CHARS = 3000


# ─── Judge helper ─────────────────────────────────────────────────────────────


def _judge_improvement(pre: str, post: str) -> int:
    """Ask Claude Haiku to rate the improvement from pre- to post-revision."""
    client = anthropic.Anthropic()
    pre_snippet = pre[:_MAX_CHARS]
    post_snippet = post[:_MAX_CHARS]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        system=JUDGE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    pre_chars=len(pre_snippet),
                    post_chars=len(post_snippet),
                    pre_revision=pre_snippet,
                    post_revision=post_snippet,
                ),
            }
        ],
    )
    raw = response.content[0].text.strip()
    try:
        score = int(raw[0])
        return max(1, min(5, score))
    except (ValueError, IndexError):
        return 1  # conservative fallback on parse failure


# ─── Eval runner ──────────────────────────────────────────────────────────────


def run_eval() -> None:
    print("=" * 60)
    print("Eval §7.3 — Critique Panel: Revision Quality")
    print(f"Runs: {RUNS}  |  Pass threshold: {PASS_THRESHOLD}/5")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Aborting.")
        return

    scores: list[int] = []

    for run_idx in range(1, RUNS + 1):
        print(f"\n--- Run {run_idx}/{RUNS} ---")
        pre_draft = PRE_REVISION_DRAFT

        # ── Step 1: critique panel ───────────────────────────────────────
        try:
            panel_state = dict(BASE_STATE)
            panel_output = critique_panel(panel_state)
            critique_warnings = panel_output.get("critique_warnings", [])
            surviving = panel_output.get("surviving_critiques", [])
            debate_rounds = len(panel_output.get("debate_log", []))
            print(
                f"  Critique panel: {len(critique_warnings)} warnings, "
                f"{len(surviving)} survived debate, {debate_rounds} debate entries"
            )
        except Exception as exc:
            print(f"  Critique panel FAILED: {exc}")
            scores.append(1)
            continue

        # ── Step 2: revision pass ────────────────────────────────────────
        try:
            revision_state = {
                **BASE_STATE,
                "latex_draft": pre_draft,
                "critique_warnings": critique_warnings,
                "surviving_critiques": surviving,
                "revision_pass_done": False,
            }
            writer_output = academic_writer(revision_state)
            post_draft = writer_output.get("latex_draft", "")
        except Exception as exc:
            print(f"  Revision pass FAILED: {exc}")
            scores.append(1)
            continue

        if not post_draft:
            print("  Revision pass returned empty draft.")
            scores.append(1)
            continue

        print(
            f"  Draft length: {len(pre_draft)} chars → {len(post_draft)} chars "
            f"(+{len(post_draft) - len(pre_draft)})"
        )

        # ── Step 3: Claude-as-judge ──────────────────────────────────────
        try:
            score = _judge_improvement(pre_draft, post_draft)
        except Exception as exc:
            print(f"  Judge call FAILED: {exc}")
            scores.append(1)
            continue

        scores.append(score)
        print(f"  Improvement score: {score}/5")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not scores:
        print("No runs completed — evaluation inconclusive.")
        return

    avg_score = sum(scores) / len(scores)
    print(f"Scores:          {scores}")
    print(f"Average score:   {avg_score:.2f}/5")
    print(f"Pass threshold:  {PASS_THRESHOLD}/5")

    if avg_score >= PASS_THRESHOLD:
        print(f"\nEvaluation PASSED  (avg {avg_score:.2f} >= {PASS_THRESHOLD})")
    else:
        print(f"\nEvaluation FAILED  (avg {avg_score:.2f} < {PASS_THRESHOLD})")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
