"""Eval: Claim Ledger No-Paper Gate.

Verifies that should_trigger_no_paper() correctly identifies when > 50%
of claims are weak or unsupported, triggering the No-Paper exit.
No API key required — pure deterministic logic.

Pass criteria: 100% correct gate decisions.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.claim_utils import should_trigger_no_paper


def _claim(strength):
    return {
        "claim_id": "c1",
        "claim_text": "test claim",
        "supporting_kg_edges": [],
        "contradicting_kg_edges": [],
        "evidence_strength": strength,
    }


# (description, ledger, expected_no_paper)
CASES = [
    ("empty ledger → no_paper (no claims to support a paper)",
     [], True),

    ("all strong → no no_paper",
     [_claim("strong"), _claim("strong"), _claim("strong")], False),

    ("all moderate → no no_paper",
     [_claim("moderate"), _claim("moderate")], False),

    ("exactly 50% weak (not strictly > 50%) → no no_paper",
     [_claim("strong"), _claim("weak")], False),

    ("51% weak (1 strong, 2 weak) → no_paper",
     [_claim("strong"), _claim("weak"), _claim("weak")], True),

    ("all weak → no_paper",
     [_claim("weak"), _claim("weak"), _claim("weak")], True),

    ("all unsupported → no_paper",
     [_claim("unsupported"), _claim("unsupported")], True),

    ("mixed: 2 strong + 3 weak = 60% weak → no_paper",
     [_claim("strong"), _claim("strong"), _claim("weak"), _claim("weak"), _claim("weak")], True),

    ("mixed: 3 strong + 2 weak = 40% weak → no no_paper",
     [_claim("strong"), _claim("strong"), _claim("strong"), _claim("weak"), _claim("weak")], False),

    ("single strong claim → no no_paper",
     [_claim("strong")], False),
]

# ─── Eval Logic ───────────────────────────────────────────────────────────────

def run_eval():
    print("Starting Claim Ledger No-Paper Gate Evaluation (no API key required)...")
    correct = 0

    for desc, ledger, expected in CASES:
        got = should_trigger_no_paper(ledger)
        ok = got == expected
        correct += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            print(f"         expected={expected}, got={got}")

    pct = correct / len(CASES) * 100
    print(f"\n--- Results ---")
    print(f"Correct: {correct}/{len(CASES)} ({pct:.0f}%)")

    if correct == len(CASES):
        print("\nEvaluation PASSED (100% correct gate decisions)")
    else:
        print(f"\nEvaluation FAILED ({correct}/{len(CASES)} — threshold is 100%)")


if __name__ == "__main__":
    run_eval()
