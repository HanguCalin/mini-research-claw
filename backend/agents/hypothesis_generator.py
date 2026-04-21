"""Node 3: Incremental Hypothesis Generator.

Type: AI (Claude Sonnet) + deterministic post-step.
Part A: LLM generates a KG-grounded hypothesis with incremental_delta.
Part B: SBERT embedding → pgvector prior-art screening → dual novelty gating.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from backend.config import MODELS, THRESHOLDS
from backend.state import AutoResearchState
from backend.utils.embeddings import embed_single
from backend.utils.supabase_client import get_supabase

logger = logging.getLogger(__name__)

HYPOTHESIS_SYSTEM_PROMPT = """\
You are a research hypothesis generator. You must produce a novel, testable
ML hypothesis grounded strictly in the Knowledge Graph provided.

RULES:
1. Every entity you mention MUST exist in the KG. Do NOT hallucinate entities.
2. Pay special attention to CONTESTED edges (supports vs contradicts between
   the same entities). Contradictions are prime targets for novel hypotheses.
3. Use ONLY real, verifiable dataset IDs from Hugging Face Hub or scikit-learn.
4. Output an "incremental_delta": 2-3 sentences explaining what is new vs
   the closest prior art.

Return ONLY valid JSON:
{
  "hypothesis": "...",
  "incremental_delta": "...",
  "mentioned_entities": ["entity_name_1", "entity_name_2"]
}
"""


def hypothesis_generator(state: AutoResearchState) -> dict[str, Any]:
    """Generate a KG-grounded hypothesis and perform novelty gating."""
    client = anthropic.Anthropic()

    kg_entities = state.get("kg_entities", [])
    kg_edges = state.get("kg_edges", [])
    topic = state.get("topic", "")

    kg_summary = _build_kg_summary(kg_entities, kg_edges)

    user_prompt = (
        f"Research topic: {topic}\n\n"
        f"Knowledge Graph:\n{kg_summary}\n\n"
        "Generate a novel, testable hypothesis grounded in this KG."
    )

    response = client.messages.create(
        model=MODELS.hypothesis_generator,
        max_tokens=2048,
        system=HYPOTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = json.loads(response.content[0].text)
    hypothesis = raw["hypothesis"]
    incremental_delta = raw.get("incremental_delta", "")
    mentioned = raw.get("mentioned_entities", [])

    entity_names = {e["canonical_name"].lower() for e in kg_entities}
    for alias_list in (e.get("aliases", []) for e in kg_entities):
        entity_names.update(a.lower() for a in alias_list)

    ungrounded = [m for m in mentioned if m.lower() not in entity_names]
    kg_valid = len(ungrounded) == 0

    if not kg_valid:
        logger.warning("Hypothesis mentions ungrounded entities: %s", ungrounded)

    hypothesis_embedding = embed_single(hypothesis)

    novelty_score, prior_art_similarity = _pgvector_novelty_check(hypothesis_embedding)

    novelty_passed = (
        novelty_score >= THRESHOLDS.novelty_threshold
        and prior_art_similarity < THRESHOLDS.prior_art_ceiling
    )

    result: dict[str, Any] = {
        "hypothesis": hypothesis,
        "incremental_delta": incremental_delta,
        "hypothesis_embedding": hypothesis_embedding,
        "novelty_score": novelty_score,
        "prior_art_similarity_score": prior_art_similarity,
        "novelty_passed": novelty_passed,
    }

    if not kg_valid:
        result["kg_valid"] = False

    if not novelty_passed:
        result["pipeline_status"] = "failed_novelty"

    return result


def _build_kg_summary(
    entities: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Render KG as a compact text block for the LLM prompt."""
    entity_map = {e["id"]: e["canonical_name"] for e in entities}

    lines: list[str] = ["Entities:"]
    for e in entities:
        lines.append(f"  - {e['canonical_name']} ({e['entity_type']})")

    lines.append("\nEdges:")
    for edge in edges:
        src = entity_map.get(edge["source_id"], "?")
        tgt = entity_map.get(edge["target_id"], "?")
        polarity = edge.get("polarity", "neutral")
        ctx = edge.get("context_condition", "")
        ctx_str = f" [condition: {ctx}]" if ctx else ""
        lines.append(
            f"  - {src} --[{edge['relation']}, {polarity}]--> {tgt}{ctx_str}"
        )

    return "\n".join(lines)


def _pgvector_novelty_check(
    hypothesis_embedding: list[float],
) -> tuple[float, float]:
    """Query Supabase pgvector for prior-art similarity and compute RND.

    Returns (novelty_score_rnd, max_prior_art_similarity).
    """
    sb = get_supabase()
    top_k = THRESHOLDS.prior_art_top_k

    response = sb.rpc(
        "match_papers",
        {
            "query_embedding": hypothesis_embedding,
            "match_count": top_k,
        },
    ).execute()

    if not response.data:
        return 1.0, 0.0

    similarities = [row["similarity"] for row in response.data]
    max_similarity = max(similarities) if similarities else 0.0

    distances = [1.0 - s for s in similarities]
    avg_distance = sum(distances) / len(distances) if distances else 1.0

    return avg_distance, max_similarity
