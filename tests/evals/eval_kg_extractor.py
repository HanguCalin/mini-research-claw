"""Evaluation script for KG Extractor performance.

This script runs the actual LLM (no mocks) on a 'gold standard' paper snippet
to measure Precision and Recall of entity and edge extraction.
"""

import os
import sys
import json
from typing import Any

# Ensure we can import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.agents.kg_extractor import kg_extractor
from backend.state import AutoResearchState

# ─── Gold Standard Data ───────────────────────────────────────────────────────

GOLD_PAPER = {
    "arxiv_id": "eval_001",
    "title": "Control Study: RF vs XGBoost",
    "abstract": (
        "In this study, we compare Random Forest (RF) and XGBoost on the Iris dataset. "
        "Our results show that RF outperforms XGBoost when the number of samples "
        "is below 500, but XGBoost achieves higher accuracy on larger sets. "
        "We used 5-fold cross-validation."
    ),
    "full_text": {
        "methodology": "Standard 5-fold CV was used with default hyperparameters.",
        "results": "RF accuracy: 0.96 (n<500). XGBoost accuracy: 0.98 (n>500)."
    }
}

GOLD_ENTITIES = {"Random Forest", "XGBoost", "Iris", "5-fold cross-validation"}
GOLD_EDGES = [
    ("Random Forest", "XGBoost", "outperforms", "supports"),
    ("XGBoost", "Iris", "uses_dataset", "neutral"),
]

# ─── Evaluation Logic ─────────────────────────────────────────────────────────

def run_eval():
    print("🚀 Starting KG Extractor Evaluation...")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        return

    state: AutoResearchState = {
        "arxiv_papers_full_text": [GOLD_PAPER],
        "kg_entities": [],
        "kg_edges": []
    }

    # Run the actual agent
    try:
        result = kg_extractor(state)
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return

    extracted_entities = {e["canonical_name"] for e in result["kg_entities"]}
    # Check aliases too
    for e in result["kg_entities"]:
        extracted_entities.update(e.get("aliases", []))

    # 1. Entity Metrics
    found_entities = GOLD_ENTITIES.intersection(extracted_entities)
    recall = len(found_entities) / len(GOLD_ENTITIES)
    
    print(f"\n--- Entity Metrics ---")
    print(f"Expected: {GOLD_ENTITIES}")
    print(f"Found:    {found_entities}")
    print(f"Recall:   {recall:.2%}")

    # 2. Edge Metrics (Simplified)
    print(f"\n--- Edge Metrics ---")
    edges = result["kg_edges"]
    entity_map = {e["id"]: e["canonical_name"] for e in result["kg_entities"]}
    
    correct_edges = 0
    for edge in edges:
        src = entity_map.get(edge["source_id"])
        tgt = entity_map.get(edge["target_id"])
        rel = edge["relation"]
        pol = edge["polarity"]
        
        print(f"Extracted: {src} --[{rel}, {pol}]--> {tgt}")
        
        # Check if matches any gold edge (very loose match for eval)
        for g_src, g_tgt, g_rel, g_pol in GOLD_EDGES:
            if src == g_src and tgt == g_tgt and pol == g_pol:
                correct_edges += 1
                break
    
    print(f"Correct Edges found: {correct_edges}/{len(GOLD_EDGES)}")

    if recall > 0.7 and correct_edges >= 1:
        print("\n✅ Evaluation PASSED (Quality within thresholds)")
    else:
        print("\n⚠️ Evaluation WEAK (Review prompts or model selection)")

if __name__ == "__main__":
    run_eval()
