"""Render the Appendix A cost-quality Pareto frontier."""

from __future__ import annotations

import argparse
from pathlib import Path


POINTS = {
    "small/direct": (1.0, 0.72),
    "small/RAG": (1.8, 0.83),
    "medium/direct": (2.8, 0.82),
    "medium/RAG": (3.6, 0.91),
    "large/agent": (7.2, 0.93),
    "large/ensemble": (11.0, 0.935),
}


def pareto_names() -> set[str]:
    """Return configurations not dominated by cheaper, no-worse points."""

    frontier: set[str] = set()
    for name, (cost, quality) in POINTS.items():
        dominated = any(
            other_cost <= cost
            and other_quality >= quality
            and (other_cost < cost or other_quality > quality)
            for other, (other_cost, other_quality) in POINTS.items()
            if other != name
        )
        if not dominated:
            frontier.add(name)
    return frontier


def plot(path: Path) -> None:
    """Write a directly labeled Pareto-frontier SVG or PNG."""

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "appendix-a-pareto"
    frontier = pareto_names()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.4, 4.1))
    for name, (cost, quality) in POINTS.items():
        is_frontier = name in frontier
        axis.scatter(cost, quality, marker="o" if is_frontier else "x", s=65, color="#2f855a" if is_frontier else "#777777")
        axis.annotate(name, (cost, quality), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ordered = sorted((POINTS[name][0], POINTS[name][1]) for name in frontier)
    axis.plot([point[0] for point in ordered], [point[1] for point in ordered], color="#2f855a", linewidth=1.5)
    axis.set(xlabel="Illustrative cost per task (USD cents)", ylabel="Task-and-policy success", ylim=(0, 1.0))
    axis.set_title("Only non-dominated configurations deserve the next review")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_format = path.suffix.removeprefix(".") or "svg"
    fig.savefig(path, format=output_format, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    plot(args.plot)
    print(sorted(pareto_names()))
