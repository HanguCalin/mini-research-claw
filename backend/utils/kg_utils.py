"""Knowledge Graph post-processing: dedup, edge resolution, incremental merge.

Used by Node 2 (KG Extractor) after LLM extraction to clean up raw entities
and edges before they enter the pipeline state.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from typing import Any

import anthropic

from backend.config import MODELS, THRESHOLDS
from backend.state import KGEdge, KGEntity
from backend.utils.embeddings import find_synonym_clusters


# ─── SBERT-based entity deduplication ────────────────────────────────────────


def deduplicate_entities_sbert(
    entities: list[KGEntity],
    client: anthropic.Anthropic,
) -> tuple[list[KGEntity], dict[str, str]]:
    """Merge entities whose canonical names are SBERT-similar > threshold.

    Returns (deduplicated_entities, id_remap) where *id_remap* maps old entity
    IDs to the surviving canonical entity ID so edges can be re-routed.
    """
    if len(entities) < 2:
        return entities, {}

    names = [e["canonical_name"] for e in entities]
    clusters = find_synonym_clusters(names, threshold=THRESHOLDS.sbert_dedup_threshold)

    id_remap: dict[str, str] = {}
    merged_ids: set[str] = set()

    for cluster_indices in clusters:
        cluster_entities = [entities[i] for i in cluster_indices]
        canonical = _llm_pick_canonical(cluster_entities, client)

        for ent in cluster_entities:
            if ent["id"] != canonical["id"]:
                id_remap[ent["id"]] = canonical["id"]
                merged_ids.add(ent["id"])

    deduped = [e for e in entities if e["id"] not in merged_ids]
    return deduped, id_remap


def _llm_pick_canonical(
    cluster: list[KGEntity],
    client: anthropic.Anthropic,
) -> KGEntity:
    """Ask Claude Haiku to pick the best canonical name for a synonym cluster."""
    cluster_json = json.dumps(
        [{"id": e["id"], "canonical_name": e["canonical_name"],
          "aliases": e["aliases"], "entity_type": e["entity_type"]}
         for e in cluster],
        indent=2,
    )
    response = client.messages.create(
        model=MODELS.kg_dedup,
        max_tokens=256,
        system=(
            "You are merging duplicate knowledge graph entities. "
            "Pick the single best canonical_name from the cluster. "
            "Merge all aliases. Return ONLY a JSON object: "
            '{"winner_id": "<id>", "merged_aliases": ["..."]}. '
            "No other text."
        ),
        messages=[{"role": "user", "content": cluster_json}],
    )
    result = json.loads(response.content[0].text)
    winner_id = result["winner_id"]

    for ent in cluster:
        if ent["id"] == winner_id:
            all_aliases: set[str] = set(ent.get("aliases", []))
            for other in cluster:
                if other["id"] != winner_id:
                    all_aliases.add(other["canonical_name"])
                    all_aliases.update(other.get("aliases", []))
            all_aliases.discard(ent["canonical_name"])
            ent["aliases"] = sorted(all_aliases)
            return ent

    return cluster[0]


# ─── Edge re-routing and dedup ───────────────────────────────────────────────


def reroute_edges(
    edges: list[KGEdge],
    id_remap: dict[str, str],
) -> list[KGEdge]:
    """Point edges at merged entity IDs after dedup."""
    result: list[KGEdge] = []
    for edge in edges:
        new_edge: dict[str, Any] = dict(edge)
        new_edge["source_id"] = id_remap.get(edge["source_id"], edge["source_id"])
        new_edge["target_id"] = id_remap.get(edge["target_id"], edge["target_id"])
        result.append(new_edge)  # type: ignore[arg-type]
    return result


def deduplicate_edges(edges: list[KGEdge]) -> list[KGEdge]:
    """Remove duplicate (source, target, relation, polarity) tuples; keep highest confidence."""
    best: dict[tuple[str, ...], KGEdge] = {}
    for edge in edges:
        key = (edge["source_id"], edge["target_id"], edge["relation"], edge["polarity"])
        if key not in best or edge["confidence"] > best[key]["confidence"]:
            best[key] = edge
    return list(best.values())


def detect_contested_pairs(edges: list[KGEdge]) -> set[tuple[str, str]]:
    """Find entity pairs that have edges with opposing polarity claims."""
    pair_polarities: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in edges:
        pair = (edge["source_id"], edge["target_id"])
        pair_polarities[pair].add(edge["polarity"])

    contested: set[tuple[str, str]] = set()
    for pair, polarities in pair_polarities.items():
        if "supports" in polarities and "contradicts" in polarities:
            contested.add(pair)
    return contested


# ─── Incremental merge ──────────────────────────────────────────────────────


def merge_kg(
    existing_entities: list[KGEntity],
    existing_edges: list[KGEdge],
    new_entities: list[KGEntity],
    new_edges: list[KGEdge],
    client: anthropic.Anthropic,
) -> tuple[list[KGEntity], list[KGEdge]]:
    """Merge new KG extractions into existing graph, dedup everything."""
    all_entities = existing_entities + new_entities
    all_edges = existing_edges + new_edges

    deduped_entities, id_remap = deduplicate_entities_sbert(all_entities, client)
    rerouted = reroute_edges(all_edges, id_remap)
    deduped_edges = deduplicate_edges(rerouted)

    return deduped_entities, deduped_edges


def make_entity_id() -> str:
    """Generate a unique entity ID."""
    return f"ent_{uuid.uuid4().hex[:12]}"


def make_edge_id() -> str:
    """Generate a unique edge ID."""
    return f"edge_{uuid.uuid4().hex[:12]}"
