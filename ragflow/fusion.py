"""Result fusion and de-duplication.

Combining multiple retrieval channels raises recall but produces duplicate and
in comparably-scaled scores. This module provides:

- min-max normalization per channel,
- Reciprocal Rank Fusion (RRF) to merge ranked lists without score calibration,
- channel-diverse de-duplication that keeps the best-scoring copy of a chunk.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from .models import Chunk


def normalize(chunks: List[Chunk]) -> List[Chunk]:
    """Min-max normalize scores in-place within a single channel."""
    if not chunks:
        return chunks
    lo = min(c.score for c in chunks)
    hi = max(c.score for c in chunks)
    span = hi - lo
    for c in chunks:
        c.score = 1.0 if span == 0 else (c.score - lo) / span
    return chunks


def rrf_fuse(lists: Iterable[List[Chunk]], k: int = 60) -> List[Chunk]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    A chunk appearing in multiple channels accumulates score, which rewards
    consensus without requiring the channels to share a score scale.
    """
    scores: Dict[str, float] = {}
    best: Dict[str, Chunk] = {}
    for chunk_list in lists:
        normalized = normalize(list(chunk_list))
        for rank, chunk in enumerate(normalized, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            existing = best.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                best[chunk.id] = chunk
    for cid, chunk in best.items():
        chunk.score = scores[cid]
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def dedupe(chunks: Iterable[Chunk]) -> List[Chunk]:
    """De-duplicate by chunk id, keeping the highest-scoring copy."""
    best: Dict[str, Chunk] = {}
    for c in chunks:
        if c.id not in best or c.score > best[c.id].score:
            best[c.id] = c
    return sorted(best.values(), key=lambda c: c.score, reverse=True)
