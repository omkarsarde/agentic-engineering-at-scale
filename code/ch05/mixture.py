"""Exact source quotas and tokenizer-fertility measurements."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable

from data_records import Document


def mix_documents(
    documents: Iterable[Document], weights: dict[str, float], *, total_docs: int
) -> list[Document]:
    """Select deterministic source quotas without silent redistribution."""

    rows = list(documents)
    if total_docs < 1 or any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("total_docs and source weights must be positive")
    positive = {source: weight for source, weight in weights.items() if weight > 0}
    eligible = [document for document in rows if document.source in positive]
    if total_docs > len(eligible):
        raise ValueError("requested mixture exceeds documents in positive-weight sources")
    total_weight = sum(positive.values())
    exact = {source: total_docs * weight / total_weight for source, weight in positive.items()}
    quotas = {source: math.floor(value) for source, value in exact.items()}
    for source in sorted(positive, key=lambda value: (-(exact[value] - quotas[value]), value)):
        if sum(quotas.values()) == total_docs:
            break
        quotas[source] += 1
    pools: dict[str, list[Document]] = defaultdict(list)
    for document in sorted(eligible, key=lambda item: item.doc_id):
        pools[document.source].append(document)
    shortfalls = {
        source: quota - len(pools[source])
        for source, quota in quotas.items()
        if len(pools[source]) < quota
    }
    if shortfalls:
        detail = ", ".join(f"{source}:{missing}" for source, missing in sorted(shortfalls.items()))
        raise ValueError(f"infeasible source quotas; missing {detail}")
    selected: list[Document] = []
    for source in sorted(quotas):
        selected.extend(pools[source][: quotas[source]])
    return selected


def measure_fertility(
    samples: dict[str, str], encode: Callable[[str], list[int]]
) -> list[dict[str, float | str]]:
    """Measure tokenizer tokens per Unicode character and whitespace word."""

    output: list[dict[str, float | str]] = []
    for language, text in samples.items():
        token_count = len(encode(text))
        output.append(
            {
                "language": language,
                "characters": float(len(text)),
                "words": float(max(1, len(text.split()))),
                "tokens": float(token_count),
                "tokens_per_character": token_count / max(1, len(text)),
                "tokens_per_word": token_count / max(1, len(text.split())),
            }
        )
    return output
