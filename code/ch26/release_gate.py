"""Uncertainty-aware release decision for the Chapter 26 mini platform."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryDecision:
    effect: str
    observed_rate: float
    lower_bound: float
    reasons: tuple[str, ...]


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    """Return a Wilson lower confidence bound for a Bernoulli rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = rate + z * z / (2 * trials)
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials**2))
    return max(0.0, (center - radius) / denominator)


def canary_gate(
    successes: int,
    trials: int,
    minimum_rate: float,
    critical_failures: int = 0,
) -> CanaryDecision:
    """Gate a canary using uncertainty plus a zero-tolerance safety rule."""
    rate = successes / trials
    lower = wilson_lower(successes, trials)
    reasons: list[str] = []
    if critical_failures:
        reasons.append("critical invariant failed")
    if lower < minimum_rate:
        reasons.append("quality lower bound misses release floor")
    effect = "PROMOTE" if not reasons else "HOLD"
    return CanaryDecision(effect, rate, lower, tuple(reasons))
