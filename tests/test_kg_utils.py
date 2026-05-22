"""Unit tests for backend/utils/kg_utils.py

Tests:
- reroute_edges remaps source/target IDs correctly
- deduplicate_edges keeps highest-confidence edge per unique key
- detect_contested_pairs identifies supports/contradicts pairs
- make_entity_id / make_edge_id generate unique IDs
"""

import sys
from unittest.mock import MagicMock

# pyarrow crashes on Python 3.14 at C-extension level; mock it and its
# dependents before any backend import triggers the load chain.
# The functions under test are pure Python and never touch these libs.
for _mod in [
    "pyarrow", "pyarrow.lib", "pyarrow.dataset",
    "sentence_transformers",
    "datasets",
    "backend.utils.embeddings",
]:
    sys.modules.setdefault(_mod, MagicMock())

import pytest

from backend.utils.kg_utils import (
    deduplicate_edges,
    detect_contested_pairs,
    make_edge_id,
    make_entity_id,
    reroute_edges,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_edge(source, target, relation="outperforms", polarity="supports",
              confidence=0.9, context_condition="", provenance="paper1"):
    return {
        "source_id": source,
        "target_id": target,
        "relation": relation,
        "polarity": polarity,
        "context_condition": context_condition,
        "confidence": confidence,
        "provenance": provenance,
    }


# ─── reroute_edges ────────────────────────────────────────────────────────────


def test_reroute_remaps_source_id():
    edges = [make_edge("old_e1", "e2")]
    result = reroute_edges(edges, {"old_e1": "new_e1"})
    assert result[0]["source_id"] == "new_e1"


def test_reroute_remaps_target_id():
    edges = [make_edge("e1", "old_e2")]
    result = reroute_edges(edges, {"old_e2": "new_e2"})
    assert result[0]["target_id"] == "new_e2"


def test_reroute_leaves_unmapped_ids_unchanged():
    edges = [make_edge("e1", "e2")]
    result = reroute_edges(edges, {"other_id": "new_id"})
    assert result[0]["source_id"] == "e1"
    assert result[0]["target_id"] == "e2"


def test_reroute_empty_remap_returns_same_edges():
    edges = [make_edge("e1", "e2"), make_edge("e3", "e4")]
    result = reroute_edges(edges, {})
    assert result[0]["source_id"] == "e1"
    assert result[1]["source_id"] == "e3"


def test_reroute_empty_edge_list():
    assert reroute_edges([], {"e1": "e2"}) == []


def test_reroute_does_not_mutate_original():
    original = make_edge("old", "e2")
    reroute_edges([original], {"old": "new"})
    assert original["source_id"] == "old"


# ─── deduplicate_edges ────────────────────────────────────────────────────────


def test_dedup_keeps_single_unique_edge():
    edges = [make_edge("e1", "e2")]
    result = deduplicate_edges(edges)
    assert len(result) == 1


def test_dedup_removes_exact_duplicate():
    edges = [make_edge("e1", "e2"), make_edge("e1", "e2")]
    result = deduplicate_edges(edges)
    assert len(result) == 1


def test_dedup_keeps_higher_confidence():
    low = make_edge("e1", "e2", confidence=0.5)
    high = make_edge("e1", "e2", confidence=0.9)
    result = deduplicate_edges([low, high])
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_dedup_keeps_both_when_polarity_differs():
    supports = make_edge("e1", "e2", polarity="supports")
    contradicts = make_edge("e1", "e2", polarity="contradicts")
    result = deduplicate_edges([supports, contradicts])
    assert len(result) == 2


def test_dedup_keeps_both_when_relation_differs():
    edge1 = make_edge("e1", "e2", relation="outperforms")
    edge2 = make_edge("e1", "e2", relation="uses_dataset")
    result = deduplicate_edges([edge1, edge2])
    assert len(result) == 2


def test_dedup_empty_list():
    assert deduplicate_edges([]) == []


# ─── detect_contested_pairs ───────────────────────────────────────────────────


def test_detects_contested_pair():
    edges = [
        make_edge("e1", "e2", polarity="supports"),
        make_edge("e1", "e2", polarity="contradicts"),
    ]
    contested = detect_contested_pairs(edges)
    assert ("e1", "e2") in contested


def test_uncontested_pair_not_flagged():
    edges = [
        make_edge("e1", "e2", polarity="supports"),
        make_edge("e1", "e2", polarity="neutral"),
    ]
    contested = detect_contested_pairs(edges)
    assert ("e1", "e2") not in contested


def test_only_supports_not_contested():
    edges = [make_edge("e1", "e2", polarity="supports")]
    assert detect_contested_pairs(edges) == set()


def test_only_contradicts_not_contested():
    edges = [make_edge("e1", "e2", polarity="contradicts")]
    assert detect_contested_pairs(edges) == set()


def test_multiple_pairs_only_contested_flagged():
    edges = [
        make_edge("e1", "e2", polarity="supports"),
        make_edge("e1", "e2", polarity="contradicts"),
        make_edge("e3", "e4", polarity="supports"),
    ]
    contested = detect_contested_pairs(edges)
    assert ("e1", "e2") in contested
    assert ("e3", "e4") not in contested


def test_empty_edge_list_returns_empty_set():
    assert detect_contested_pairs([]) == set()


# ─── make_entity_id / make_edge_id ───────────────────────────────────────────


def test_make_entity_id_starts_with_prefix():
    assert make_entity_id().startswith("ent_")


def test_make_edge_id_starts_with_prefix():
    assert make_edge_id().startswith("edge_")


def test_make_entity_id_is_unique():
    ids = {make_entity_id() for _ in range(100)}
    assert len(ids) == 100


def test_make_edge_id_is_unique():
    ids = {make_edge_id() for _ in range(100)}
    assert len(ids) == 100
