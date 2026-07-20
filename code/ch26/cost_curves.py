"""Generate the quantitative cache and canary figure for Chapter 26."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def cached_cost(
    reuses: int, uncached_cost: float, write_multiplier: float, read_multiplier: float
) -> float:
    """Price one cache write followed by `reuses` discounted reads."""
    return uncached_cost * (write_multiplier + reuses * read_multiplier)


def uncached_cost(requests: int, unit_cost: float) -> float:
    return requests * unit_cost


def detection_probability(sample_size: int, regression_rate: float) -> float:
    """Probability of observing at least one independently regressed outcome."""
    return 1 - (1 - regression_rate) ** sample_size


def plot(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-26"
    path.parent.mkdir(parents=True, exist_ok=True)
    reuses = list(range(0, 21))
    savings = [
        uncached_cost(r + 1, 1.0) - cached_cost(r, 1.0, 1.25, 0.10)
        for r in reuses
    ]
    samples = list(range(1, 101))
    powers = [detection_probability(n, 0.05) for n in samples]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].axhline(0, color="#666666", linewidth=1)
    axes[0].plot(reuses, savings, color="#315b8a", linewidth=2)
    axes[0].set(xlabel="Cache reuses", ylabel="Savings (uncached cost units)", title="Prefix-cache break-even")
    axes[1].plot(samples, powers, color="#2f855a", linewidth=2)
    axes[1].axhline(0.8, color="#666666", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Independent canary tasks", ylabel="Detection probability", title="Detect one 5% regression")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    plot(args.plot)
    break_even = next(
        r
        for r in range(100)
        if cached_cost(r, 1.0, 1.25, 0.10) <= uncached_cost(r + 1, 1.0)
    )
    n80 = math.ceil(math.log(1 - 0.80) / math.log(1 - 0.05))
    print({"cache_reuses_to_break_even": break_even, "tasks_for_80pct_detection": n80})
