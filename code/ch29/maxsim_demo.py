"""Contrast pooled retrieval with late interaction on one transparent example."""

from __future__ import annotations

import json
import math
from typing import Iterable


Vector = tuple[float, ...]


def normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def pooled(vectors: list[Vector]) -> Vector:
    return normalize(tuple(sum(values) / len(vectors) for values in zip(*vectors)))


def pooled_score(query: list[Vector], document: list[Vector]) -> float:
    return dot(pooled(query), pooled(document))


def maxsim(query: list[Vector], document: list[Vector]) -> float:
    """Give every query token its best matching document patch."""
    return sum(max(dot(normalize(q), normalize(d)) for d in document) for q in query)


def run_demo() -> dict[str, dict[str, float]]:
    query = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    relevant = [query[0], query[1]] + [(0.0, 0.0, 1.0)] * 8
    flooded = [(0.7, 0.7, 0.0)] * 10
    return {
        "pooled": {
            "relevant": pooled_score(query, relevant),
            "flooded": pooled_score(query, flooded),
        },
        "maxsim": {
            "relevant": maxsim(query, relevant),
            "flooded": maxsim(query, flooded),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
