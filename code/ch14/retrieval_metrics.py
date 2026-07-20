"""Ranked retrieval metrics with explicit duplicate-source handling."""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Measure the fraction of required source identities found by rank k."""
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant source."""
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at k with duplicate source hits ignored."""
    if not relevant:
        return 1.0
    seen: set[str] = set()
    total = hits = 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant and item not in seen:
            seen.add(item)
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Compute normalized discounted cumulative gain at k."""
    dcg = sum((2 ** gains.get(item, 0.0) - 1) / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg else 1.0
