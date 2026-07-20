"""Render success and cost against capstone complexity rung."""

from __future__ import annotations

import argparse
from pathlib import Path

from readiness_ledger import fixture_ladder


def plot(path: Path) -> None:
    """Write the Chapter 32 ablation decision plot."""

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-32"
    reports = fixture_ladder()
    x = [report.rung for report in reports]
    success = [100 * report.task_success for report in reports]
    cost = [1000 * report.cost_per_task_usd for report in reports]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharex=True)
    axes[0].plot(x, success, "o-", color="#2f855a")
    axes[0].set(ylabel="Golden-set success (%)", ylim=(0, 100))
    axes[1].plot(x, cost, "s-", color="#315b8a")
    axes[1].set(ylabel="Cost per task (USD mills)")
    for axis in axes:
        axis.set_xlabel("Architecture rung")
        axis.set_xticks(x)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.axvspan(4.75, 6.25, color="#b13f3f", alpha=0.08)
    axes[0].annotate("no measured gain", xy=(5.5, 95), xytext=(4.25, 87), arrowprops={"arrowstyle": "->"})
    axes[1].annotate("cost more than doubles", xy=(5, 25), xytext=(2.7, 35), arrowprops={"arrowstyle": "->"})
    fig.suptitle("The adaptive loop and memory fail to earn their cost in this fixture")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_format = path.suffix.removeprefix(".") or "svg"
    fig.savefig(path, format=output_format, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    plot(args.plot)
