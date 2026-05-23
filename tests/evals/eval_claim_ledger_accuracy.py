"""Eval: Claim Ledger Evidence Strength Accuracy.

Injects KG edges with known support/contradiction patterns and verifies
that rate_evidence_strength produces correct ratings.
No API key required — claim ledger builder is fully deterministic.

Pass criteria: >= 90% of ratings correct.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.claim_utils import rate_evidence_strength

# ─── Test cases: (supporting_edges, contradicting_edges, unconditional, expected_strength) ──

def _make_edge(polarity="supports", context_condition=""):
    return {
        "source_id": "e1", "target_id": "e2",
        "relation": "outperforms", "polarity": polarity,
        "context_condition": context_condition,
        "confidence": 0.9, "provenance": "p1/results",
    }


CASES = [
    # (description, supporting, contradicting, claim_is_unconditional, expected)
    ("2 supporting, 0 contra → strong",
     [_make_edge(), _make_edge()], [], True, "strong"),

    ("1 supporting, 0 contra → moderate",
     [_make_edge()], [], True, "moderate"),

    ("3 supporting, 1 contra → moderate",
     [_make_edge(), _make_edge(), _make_edge()], [_make_edge("contradicts")], True, "moderate"),

    ("1 supporting, 1 contra → weak",
     [_make_edge()], [_make_edge("contradicts")], True, "weak"),

    ("0 supporting, 2 contra → unsupported",
     [], [_make_edge("contradicts"), _make_edge("contradicts")], True, "unsupported"),

    ("0 edges → unsupported",
     [], [], True, "unsupported"),

    ("conditional edge for unconditional claim (0.5 effective) → unsupported",
     [_make_edge(context_condition="only when n<500")], [], True, "unsupported"),

    ("2 conditional edges for unconditional claim (1.0 effective) → moderate",
     [_make_edge(context_condition="only n<500"), _make_edge(context_condition="only n<500")],
     [], True, "moderate"),

    ("1 conditional edge, claim IS conditional → moderate (full credit)",
     [_make_edge(context_condition="only when n<500")], [], False, "moderate"),

    ("2 supporting + 0 contra, unconditional edges → strong (high confidence path)",
     [_make_edge(), _make_edge(), _make_edge()], [], True, "strong"),
]

# ─── Eval Logic ───────────────────────────────────────────────────────────────

def run_eval():
    print("Starting Claim Ledger Accuracy Evaluation (no API key required)...")
    correct = 0

    for desc, supporting, contradicting, unconditional, expected in CASES:
        got = rate_evidence_strength(supporting, contradicting, unconditional)
        ok = got == expected
        correct += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         expected={expected!r}, got={got!r}")

    pct = correct / len(CASES) * 100
    print(f"\n--- Results ---")
    print(f"Correct: {correct}/{len(CASES)} ({pct:.0f}%)")

    if pct >= 90:
        print("\nEvaluation PASSED (>= 90% ratings correct)")
    else:
        print(f"\nEvaluation FAILED ({pct:.0f}% — threshold is 90%)")


if __name__ == "__main__":
    run_eval()
