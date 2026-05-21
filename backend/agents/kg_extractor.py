"""Node 2: Epistemic KG Extractor.

Type: AI (Claude Haiku).
Iterates over papers one at a time. For each paper, calls Claude with a
schema-based prompt to extract typed entities and epistemic edges (with
polarity, context_condition, confidence, provenance). Then runs the
deterministic post-processing pipeline (SBERT dedup, LLM dedup, edge
resolution, contested-pair detection).
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic

from backend.config import MODELS
from backend.state import AutoResearchState, KGEdge, KGEntity
from backend.utils.llm_utils import extract_json, extract_text
from backend.utils.kg_utils import (
    deduplicate_edges,
    deduplicate_entities_sbert,
    detect_contested_pairs,
    make_edge_id,
    make_entity_id,
    merge_kg,
    reroute_edges,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge graph extractor for academic ML papers.

Given a paper's text, extract entities and edges into strict JSON.

Entity types: model, dataset, metric, method, hyperparameter.
Relation types: outperforms, uses_dataset, achieves_metric, has_hyperparameter.

CRITICAL RULES:
1. Every edge MUST have a "polarity" field: "supports", "contradicts", or "neutral".
2. Every edge MUST have a "context_condition" field. If the finding is conditional
   (e.g., "RF outperforms XGBoost on small datasets"), capture the boundary in
   context_condition. If unconditional, use an empty string "".
3. NEVER treat conditional support as absolute.
4. Every edge MUST have "confidence" (float 0-1) and "provenance" (paper ID + section).

Return ONLY valid JSON with this exact schema — no free-form text:
{
  "entities": [
    {
      "canonical_name": "...",
      "aliases": ["..."],
      "entity_type": "model|dataset|metric|method|hyperparameter",
      "attributes": {}
    }
  ],
  "edges": [
    {
      "source_name": "...",
      "target_name": "...",
      "relation": "outperforms|uses_dataset|achieves_metric|has_hyperparameter",
      "polarity": "supports|contradicts|neutral",
      "context_condition": "",
      "confidence": 0.0
    }
  ]
}
"""


def kg_extractor(state: AutoResearchState) -> dict[str, Any]:
    """Extract KG entities and edges from all papers, then dedup and merge."""
    papers = state.get("arxiv_papers_full_text", [])
    existing_entities: list[KGEntity] = list(state.get("kg_entities", []))
    existing_edges: list[KGEdge] = list(state.get("kg_edges", []))

    client = anthropic.Anthropic()
    new_entities: list[KGEntity] = []
    new_edges: list[KGEdge] = []

    for paper in papers:
        paper_id = paper.get("arxiv_id", "unknown")
        paper_text = _format_paper_for_prompt(paper)

        try:
            response = client.messages.create(
                model=MODELS.kg_extractor,
                max_tokens=4096,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": paper_text}],
            )
            raw = extract_json(extract_text(response))
        except (ValueError, anthropic.APIError) as exc:
            logger.warning("KG extraction failed for %s: %s", paper_id, exc)
            continue

        name_to_id: dict[str, str] = {}

        for raw_ent in raw.get("entities", []):
            eid = make_entity_id()
            name_to_id[raw_ent["canonical_name"]] = eid
            new_entities.append(KGEntity(
                id=eid,
                canonical_name=raw_ent["canonical_name"],
                aliases=raw_ent.get("aliases", []),
                entity_type=raw_ent.get("entity_type", "method"),
                attributes=raw_ent.get("attributes", {}),
            ))

        for raw_edge in raw.get("edges", []):
            src = name_to_id.get(raw_edge["source_name"], "")
            tgt = name_to_id.get(raw_edge["target_name"], "")
            if not src or not tgt:
                continue

            new_edges.append(KGEdge(
                source_id=src,
                target_id=tgt,
                relation=raw_edge.get("relation", "outperforms"),
                polarity=raw_edge.get("polarity", "neutral"),
                context_condition=raw_edge.get("context_condition", ""),
                confidence=float(raw_edge.get("confidence", 0.5)),
                provenance=paper_id,
            ))

    merged_entities, merged_edges = merge_kg(
        existing_entities, existing_edges,
        new_entities, new_edges,
        client,
    )

    contested = detect_contested_pairs(merged_edges)
    if contested:
        logger.info("Found %d contested entity pairs", len(contested))

    return {
        "kg_entities": merged_entities,
        "kg_edges": merged_edges,
    }


def _format_paper_for_prompt(paper: dict[str, Any]) -> str:
    """Format a single paper dict into a compact text block for the LLM."""
    parts = [f"Title: {paper.get('title', 'N/A')}"]
    parts.append(f"ArXiv ID: {paper.get('arxiv_id', 'N/A')}")

    abstract = paper.get("abstract", "")
    if abstract:
        parts.append(f"Abstract: {abstract}")

    full_text = paper.get("full_text", {})
    if isinstance(full_text, dict):
        for section, content in full_text.items():
            if content:
                parts.append(f"[{section}]: {content}")

    return "\n\n".join(parts)
