"""Repeated-run estimators and paired task-cluster uncertainty."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist, fmean, stdev
from typing import Any


def pass_at_k_estimate(n: int, successes: int, k: int) -> float:
    """Estimate the chance that at least one of k sampled runs succeeds."""
    if not 1 <= k <= n or not 0 <= successes <= n:
        raise ValueError("require 1 <= k <= n and 0 <= successes <= n")
    failures = n - successes
    return 1.0 if failures < k else 1.0 - math.comb(failures, k) / math.comb(n, k)


def pass_pow_k_estimate(n: int, successes: int, k: int) -> float:
    """Estimate the chance that all k sampled runs succeeds."""
    if not 1 <= k <= n or not 0 <= successes <= n:
        raise ValueError("require 1 <= k <= n and 0 <= successes <= n")
    return 0.0 if successes < k else math.comb(successes, k) / math.comb(n, k)


def task_metrics(rows: list[dict[str, Any]], system: str, k: int = 2) -> dict[str, dict[str, float]]:
    """Aggregate repeated trials within each independent task cluster."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[row["task_id"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for task_id, trials in sorted(grouped.items()):
        n = len(trials)
        successes = sum(row["success"] for row in trials)
        result[task_id] = {
            "pass_rate": successes / n,
            "pass_at_k": pass_at_k_estimate(n, successes, k),
            "pass_pow_k": pass_pow_k_estimate(n, successes, k),
            "trajectory_f1": fmean(row["trajectory_f1"] for row in trials),
        }
    return result


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_cluster_uncertainty(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    *,
    resamples: int = 10_000,
    seed: int = 22,
) -> dict[str, Any]:
    """Bootstrap paired task clusters and report an approximate 80% MDE."""
    if set(baseline) != set(candidate) or len(baseline) < 2:
        raise ValueError("paired systems need the same two or more task IDs")
    deltas = [candidate[key]["pass_rate"] - baseline[key]["pass_rate"] for key in sorted(baseline)]
    rng = random.Random(seed)
    boot = [fmean(rng.choice(deltas) for _ in deltas) for _ in range(resamples)]
    sigma = stdev(deltas)
    cluster_se = sigma / math.sqrt(len(deltas))
    z_alpha = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.80)
    return {
        "point": fmean(deltas),
        "low": percentile(boot, 0.025),
        "high": percentile(boot, 0.975),
        "cluster_se": cluster_se,
        "mde_80": (z_alpha + z_power) * cluster_se,
        "task_deltas": dict(zip(sorted(baseline), deltas)),
    }


def slice_rates(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    """Return trial success rates for predeclared workload slices."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[row["slice"]].append(row["success"])
    return {name: fmean(values) for name, values in sorted(grouped.items())}
