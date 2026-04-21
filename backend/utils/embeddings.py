"""SBERT embedding utilities (all-MiniLM-L6-v2, 384 dimensions).

Used by:
  - Node 1 (ArXiv Retriever) — embed papers for cache insert
  - Node 2 (KG Extractor)    — entity deduplication via cosine similarity
  - Node 3 (Hypothesis Gen)  — embed hypothesis for novelty detection
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import SBERT_EMBEDDING_DIM, SBERT_MODEL_NAME


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(SBERT_MODEL_NAME)


def embed_texts(texts: Sequence[str]) -> NDArray[np.float32]:
    """Encode *texts* → (N, 384) float32 matrix."""
    model = _get_model()
    vecs = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    assert vecs.shape[1] == SBERT_EMBEDDING_DIM
    return vecs.astype(np.float32)


def embed_single(text: str) -> list[float]:
    """Encode a single string → 384-dim list[float] (Supabase-compatible)."""
    return embed_texts([text])[0].tolist()


def pairwise_cosine(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the (N, N) cosine-similarity matrix."""
    return cosine_similarity(matrix).astype(np.float32)


def find_synonym_clusters(
    names: list[str],
    threshold: float = 0.85,
) -> list[list[int]]:
    """Group *names* indices where pairwise cosine similarity > *threshold*.

    Returns a list of clusters; each cluster is a list of indices into *names*.
    Singletons are omitted.
    """
    if len(names) < 2:
        return []

    vecs = embed_texts(names)
    sim = pairwise_cosine(vecs)

    visited: set[int] = set()
    clusters: list[list[int]] = []

    for i in range(len(names)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, len(names)):
            if j not in visited and sim[i, j] > threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters
