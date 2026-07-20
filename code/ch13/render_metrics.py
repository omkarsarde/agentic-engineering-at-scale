"""Render Chapter 13's 50-turn context ledger as a print-safe SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

from context_pipeline import run_build


def render(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "ch13-context-ops"
    rows = run_build()["ledger"]
    scenarios = sorted({row["scenario"] for row in rows})
    colors = {
        "compaction=off, prefix=broken": "#a84f45",
        "compaction=off, prefix=stable": "#d58b36",
        "compaction=on, prefix=broken": "#64748b",
        "compaction=on, prefix=stable": "#2e6f57",
    }
    styles = {name: ("-" if "prefix=stable" in name else "--") for name in scenarios}
    markers = {name: ("o" if "compaction=on" in name else "s") for name in scenarios}
    short_labels = {
        "compaction=off, prefix=broken": "off · broken",
        "compaction=off, prefix=stable": "off · stable",
        "compaction=on, prefix=broken": "on · broken",
        "compaction=on, prefix=stable": "on · stable",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for scenario in scenarios:
        selected = [row for row in rows if row["scenario"] == scenario]
        turns = [row["turn"] for row in selected]
        utilization = [row["utilization"] for row in selected]
        cache_rate = [row["cache_hit_rate"] for row in selected]
        for axis, values in zip(axes, (utilization, cache_rate), strict=True):
            axis.plot(
                turns,
                values,
                linestyle=styles[scenario],
                marker=markers[scenario],
                markevery=5,
                markersize=4,
                color=colors[scenario],
                linewidth=2,
            )

    axes[0].axhline(0.65, color="#222222", linewidth=1, alpha=0.55)
    axes[0].text(2, 0.67, "65% compaction trigger", fontsize=8, va="bottom")
    axes[0].set_title("Context utilization")
    axes[0].set_ylabel("Whitespace-token proxy / budget")
    axes[1].set_title("Reusable leading-byte fraction")
    axes[1].set_ylabel("Common prefix bytes / request bytes")
    for axis in axes:
        axis.set_xlabel("Turn")
        axis.set_xlim(1, 61)
        axis.set_ylim(0, 1.12)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    endpoint_rows = {
        scenario: [row for row in rows if row["scenario"] == scenario][-1]
        for scenario in scenarios
    }
    label_positions = (
        {
            "compaction=off, prefix=broken": 1.08,
            "compaction=off, prefix=stable": 0.98,
            "compaction=on, prefix=broken": 0.69,
            "compaction=on, prefix=stable": 0.58,
        },
        {
            "compaction=off, prefix=broken": 0.08,
            "compaction=off, prefix=stable": 1.05,
            "compaction=on, prefix=broken": 0.17,
            "compaction=on, prefix=stable": 0.94,
        },
    )
    value_keys = ("utilization", "cache_hit_rate")
    for axis, positions, value_key in zip(axes, label_positions, value_keys, strict=True):
        for scenario in scenarios:
            axis.annotate(
                short_labels[scenario],
                xy=(50, endpoint_rows[scenario][value_key]),
                xytext=(51.5, positions[scenario]),
                fontsize=7.5,
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.65},
            )
    fig.suptitle("Compaction and prefix stability solve different constraints", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
