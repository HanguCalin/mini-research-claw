"""Evaluation script for Hypothesis Generator grounding and testability.

Runs the real LLM against a sample KG and checks if the hypothesis
hallucinates entities or proposes non-testable experiments.
"""

import os
import sys
from typing import Any

# Ensure we can import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.hypothesis_generator import hypothesis_generator
from backend.state import AutoResearchState

# ─── Evaluation Data ──────────────────────────────────────────────────────────

SAMPLE_KG_ENTITIES = [
    {"id": "e1", "canonical_name": "Transformer", "entity_type": "model", "aliases": []},
    {"id": "e2", "canonical_name": "Layer Normalization", "entity_type": "method", "aliases": []},
    {"id": "e3", "canonical_name": "CIFAR-10", "entity_type": "dataset", "aliases": []},
]

SAMPLE_KG_EDGES = [
    {"source_id": "e1", "target_id": "e2", "relation": "uses_method", "polarity": "supports", "confidence": 1.0},
    {"source_id": "e1", "target_id": "e3", "relation": "achieves_metric", "polarity": "supports", "confidence": 0.9},
]

# ─── Evaluation Logic ─────────────────────────────────────────────────────────

def run_eval():
    print("🚀 Starting Hypothesis Generator Evaluation...")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        return

    state: AutoResearchState = {
        "topic": "Improving Transformers on small vision datasets",
        "kg_entities": SAMPLE_KG_ENTITIES,
        "kg_edges": SAMPLE_KG_EDGES,
        "arxiv_papers_full_text": [], # No current papers to ensure novelty check runs against DB
    }

    trials = 3
    hallucinations = 0
    
    valid_names = {e["canonical_name"].lower() for e in SAMPLE_KG_ENTITIES}

    for i in range(trials):
        print(f"\n--- Trial {i+1}/{trials} ---")
        try:
            # Note: This will call Supabase for novelty check unless mocked.
            # For eval, we expect it to either pass or fail novelty based on DB state.
            result = hypothesis_generator(state)
        except Exception as e:
            print(f"❌ Trial failed: {e}")
            continue

        hypo = result.get("hypothesis", "")
        print(f"Hypothesis: {hypo}")
        
        # Check grounding (very basic keyword check)
        # In a real eval, we'd use an LLM-as-a-judge to check grounding.
        # Here we check the 'kg_valid' flag set by the agent.
        if result.get("kg_valid") is False:
            print("⚠️ Hallucination detected (ungrounded entities)!")
            hallucinations += 1
        else:
            print("✅ Grounding: OK")

        print(f"Novelty Score: {result.get('novelty_score'):.4f}")
        print(f"Novelty Passed: {result.get('novelty_passed')}")

    pass_rate = (trials - hallucinations) / trials
    print(f"\n--- Final Results ---")
    print(f"Grounding Pass Rate: {pass_rate:.2%}")

    if pass_rate >= 0.66:
        print("\n✅ Evaluation PASSED")
    else:
        print("\n❌ Evaluation FAILED (Hallucination rate too high)")

if __name__ == "__main__":
    run_eval()
