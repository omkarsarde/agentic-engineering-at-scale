"""Render the error-budget burn figure for Chapter 27."""

from __future__ import annotations

import argparse
from pathlib import Path


def remaining(day: float, burn: float, window_days: float = 30.0) -> float:
    return max(0.0, 1 - burn * day / window_days)


def plot(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-27"
    path.parent.mkdir(parents=True, exist_ok=True)
    days = [value / 10 for value in range(0, 301)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for burn, style in [(1.0, "-"), (6.9, "--"), (14.4, ":")]:
        ax.plot(days, [remaining(day, burn) for day in days], style, linewidth=2, label=f"{burn:g}× burn")
    ax.set(xlabel="Days at constant burn", ylabel="Error budget remaining", title="Burn rate converts a green percentage into urgency")
    ax.set(xlim=(0, 30), ylim=(0, 1))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", required=True, type=Path)
    args = parser.parse_args()
    plot(args.plot)
    print({"days_at_6.9x_with_95pct_remaining": round(30 * 0.95 / 6.9, 1)})
