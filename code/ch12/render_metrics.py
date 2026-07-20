"""Render the deterministic structured-output results as a print-safe SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

from chapter_build import run


def render(path: Path) -> None:
    """Plot validity by guarantee and expose the semantic-invalid remainder."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-12"
    report = run()
    names = ["Prompt only", "JSON mode", "Schema constrained", "App validated"]
    rates = [
        report["schema_validity"]["level_1"],
        report["schema_validity"]["level_2"],
        report["schema_validity"]["level_3"],
        1.0,
    ]
    invalid = [0.0, 0.0, report["layer3_business_invalid"] / report["tickets"], 0.0]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    positions = range(len(names))
    ax.bar(positions, rates, color=["#d7dde5", "#aebac9", "#71849c", "#355f4d"], edgecolor="#222222")
    ax.bar(positions, invalid, bottom=[rate - bad for rate, bad in zip(rates, invalid)], color="none", edgecolor="#111111", hatch="////", linewidth=1.2, label="Schema-valid, business-invalid")
    for position, rate in zip(positions, rates):
        ax.text(position, rate + 0.025, f"{rate:.0%}", ha="center", fontsize=9)
    ax.set_xticks(list(positions), names)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Valid outputs / 50 tickets")
    ax.set_title("Stronger decoding constraints improve shape, not business truth")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
