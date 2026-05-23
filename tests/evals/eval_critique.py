"""Evaluation script for the Critique Panel agents.

Feeds a deliberately 'bad' paper draft to the panel to verify that
the agents (Fact Checker, Methodologist, Formatter) identify flaws.
"""

import os
import sys
import json

# Ensure we can import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.critique_panel import critique_panel
from backend.state import AutoResearchState

# ─── 'Bad' Paper Data ─────────────────────────────────────────────────────────

BAD_STATE = {
    "topic": "Neural Networks for Iris",
    "hypothesis": "A 1-layer perceptron is enough.",
    "latex_draft": (
        "\\section{Intro} This paper is very short. "
        "\\section{Methods} We did some things. "
        "No details on cross-validation or hyperparameters."
    ),
    "bibtex_source": "@article{fake_ref, author={Unknown}, title={Empty}}",
    "metrics_json": {"accuracy": 0.5}, # Low accuracy
    "claim_ledger": [
        {"claim_id": "c1", "claim_text": "We found magic results.", "evidence_strength": "weak"}
    ],
    "kg_entities": [],
    "kg_edges": [],
    "critique_warnings": [],
}

# ─── Evaluation Logic ─────────────────────────────────────────────────────────

def run_eval():
    print("🚀 Starting Critique Panel Evaluation (Negative Test)...")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        return

    try:
        # Run the real panel (3 agents + debate)
        result = critique_panel(BAD_STATE)
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return

    warnings = result.get("critique_warnings", [])
    surviving = result.get("surviving_critiques", [])
    debate = result.get("debate_log", [])

    print(f"\n--- Results ---")
    print(f"Total warnings generated: {len(warnings)}")
    print(f"Critiques surviving debate: {len(surviving)}")
    print(f"Debate rounds occurred: {len(debate)}")

    # Check if agents caught the lack of methodology
    found_method_critique = False
    for c in surviving:
        text = c.get("critique", "").lower()
        print(f"  - [{c.get('source')}]: {c.get('critique')}")
        if "method" in text or "detail" in text or "validation" in text:
            found_method_critique = True

    if len(surviving) >= 2 and found_method_critique:
        print("\n✅ Evaluation PASSED (Agents identified major flaws)")
    else:
        print("\n❌ Evaluation FAILED (Agents were too lenient)")

if __name__ == "__main__":
    run_eval()
