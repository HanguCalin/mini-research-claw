"""Node 3c: Experimental Designer.

Type: AI (Claude Sonnet).
Generates a structured ExperimentSpec from the approved hypothesis and KG.
All 6 required fields must be present and non-empty.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from backend.config import MODELS
from backend.state import AutoResearchState, ExperimentSpec
from backend.utils.llm_utils import extract_json, extract_text

logger = logging.getLogger(__name__)

DESIGNER_SYSTEM_PROMPT = """\
You are an experimental designer for ML research. Given a hypothesis and
knowledge graph context, design a rigorous experiment.

RULES:
1. The dataset_id MUST be a real, publicly available dataset from Hugging Face
   Hub (e.g., "imdb", "glue/sst2") or scikit-learn (e.g., "sklearn.iris").
2. evaluation_metrics must be concrete, measurable metrics (accuracy, F1, etc.).
3. Justify each choice in 1-2 sentences.

Return ONLY valid JSON with this exact schema:
{
  "independent_var": "...",
  "dependent_var": "...",
  "control_description": "...",
  "dataset_id": "...",
  "evaluation_metrics": ["metric1", "metric2"],
  "expected_outcome": "...",
  "justifications": {
    "independent_var": "...",
    "dependent_var": "...",
    "control_description": "...",
    "dataset_id": "...",
    "evaluation_metrics": "...",
    "expected_outcome": "..."
  }
}
"""

REQUIRED_FIELDS = [
    "independent_var", "dependent_var", "control_description",
    "dataset_id", "evaluation_metrics", "expected_outcome",
]


def experiment_designer(state: AutoResearchState) -> dict[str, Any]:
    """Design an experiment spec from the hypothesis and KG context."""
    client = anthropic.Anthropic()

    hypothesis = state.get("hypothesis", "")
    incremental_delta = state.get("incremental_delta", "")
    kg_entities = state.get("kg_entities", [])
    kg_edges = state.get("kg_edges", [])

    entity_summary = ", ".join(e["canonical_name"] for e in kg_entities[:20])
    edge_summary = "; ".join(
        f"{e['source_id']}→{e['target_id']} ({e['relation']}, {e['polarity']})"
        for e in kg_edges[:15]
    )

    user_prompt = (
        f"Hypothesis: {hypothesis}\n\n"
        f"What's new: {incremental_delta}\n\n"
        f"KG Entities: {entity_summary}\n\n"
        f"KG Edges: {edge_summary}\n\n"
        "Design a rigorous experiment to test this hypothesis."
    )

    response = client.messages.create(
        model=MODELS.experiment_designer,
        max_tokens=2048,
        system=DESIGNER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = extract_json(extract_text(response))

    missing = [f for f in REQUIRED_FIELDS if not raw.get(f)]
    if missing:
        raise ValueError(f"ExperimentSpec missing required fields: {missing}")

    metrics = raw["evaluation_metrics"]
    if isinstance(metrics, str):
        metrics = [m.strip() for m in metrics.split(",")]

    spec = ExperimentSpec(
        independent_var=raw["independent_var"],
        dependent_var=raw["dependent_var"],
        control_description=raw["control_description"],
        dataset_id=raw["dataset_id"],
        evaluation_metrics=metrics,
        expected_outcome=raw["expected_outcome"],
    )

    return {"experiment_spec": spec}
