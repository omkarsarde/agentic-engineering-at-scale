"""Logit processing, sampling, and streaming for the inference lab."""

from __future__ import annotations

import codecs
import math
import random


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Return a stable temperature-scaled categorical distribution."""
    if temperature <= 0:
        winner = max(range(len(logits)), key=logits.__getitem__)
        return [float(i == winner) for i in range(len(logits))]
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def process_logits(
    logits: list[float], counts: list[int], repetition: float = 1.0,
    frequency: float = 0.0, presence: float = 0.0,
    bias: dict[int, float] | None = None,
) -> list[float]:
    """Apply sign-aware repetition, additive penalties, and explicit bias."""
    result = logits[:]
    for i, count in enumerate(counts):
        if count and repetition != 1.0:
            result[i] = result[i] / repetition if result[i] > 0 else result[i] * repetition
        result[i] -= frequency * count + presence * float(count > 0)
        result[i] += (bias or {}).get(i, 0.0)
    return result


def truncate(probs: list[float], method: str = "full", value: float = 1.0) -> list[float]:
    """Mask and renormalize with top-k, top-p, min-p, or typical sampling."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    if method == "full":
        keep = set(order)
    elif method == "top_k":
        keep = set(order[: max(1, int(value))])
    elif method == "top_p":
        keep, mass = set(), 0.0
        for i in order:
            keep.add(i)
            mass += probs[i]
            if mass >= value:
                break
    elif method == "min_p":
        cutoff = value * probs[order[0]]
        keep = {i for i in order if probs[i] >= cutoff} or {order[0]}
    elif method == "typical":
        entropy = -sum(p * math.log(p) for p in probs if p)
        typical = sorted(order, key=lambda i: abs(-math.log(probs[i]) - entropy))
        keep, mass = set(), 0.0
        for i in typical:
            keep.add(i)
            mass += probs[i]
            if mass >= value:
                break
    else:
        raise ValueError(f"unknown truncation method: {method}")
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]


def distribution(logits: list[float], temperature: float, method: str, value: float) -> list[float]:
    """Run temperature, truncation, and renormalization stages."""
    return truncate(softmax(logits, temperature), method, value)


def sample_index(probs: list[float], rng: random.Random) -> int:
    """Draw one categorical outcome using only the supplied RNG."""
    draw, cumulative = rng.random(), 0.0
    for i, probability in enumerate(probs):
        cumulative += probability
        if draw <= cumulative:
            return i
    return len(probs) - 1


def sampling_probe() -> list[dict[str, float | int | str]]:
    """Compare support and entropy under five decoding policies."""
    logits = [4.0, 3.2, 2.2, 1.8, 0.0]
    policies = [("full", 1.0), ("top_k", 2), ("top_p", 0.8), ("min_p", 0.15), ("typical", 0.8)]
    rows = []
    for method, value in policies:
        probs = distribution(logits, 1.2, method, value)
        draws = [sample_index(probs, random.Random(seed)) for seed in range(200)]
        rows.append({"policy": method, "support": sum(p > 0 for p in probs),
                     "distinct_200": len(set(draws)),
                     "entropy_nats": -sum(p * math.log(p) for p in probs if p)})
    return rows


def stream_until(chunks: list[bytes], stop: str) -> str:
    """Decode partial UTF-8 safely while matching stops across chunk boundaries."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending, emitted = "", []
    for chunk in chunks:
        pending += decoder.decode(chunk)
        if stop in pending:
            emitted.append(pending.split(stop, 1)[0])
            return "".join(emitted)
        keep = max(0, len(stop) - 1)
        if len(pending) > keep:
            emitted.append(pending[:-keep] if keep else pending)
            pending = pending[-keep:] if keep else ""
    pending += decoder.decode(b"", final=True)
    return "".join(emitted) + pending
