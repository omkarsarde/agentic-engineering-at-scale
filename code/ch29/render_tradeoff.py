"""Render field accuracy and image-token cost for the Chapter 29 sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from page_reader import run_sweep


def plot(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-29"
    report = run_sweep()
    tiles = [row["tiles"] for row in report]
    accuracy = [row["field_accuracy"] for row in report]
    tokens = [row["image_tokens_per_page"] for row in report]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
    ax2 = ax1.twinx()
    first = ax1.plot(tiles, accuracy, "o-", color="#2f855a", label="Field accuracy")
    second = ax2.plot(tiles, tokens, "s--", color="#315b8a", label="Image tokens")
    ax1.set(xlabel="Image tiles per page", ylabel="Exact field accuracy", ylim=(0, 1.05))
    ax2.set_ylabel("Illustrative image tokens per page")
    ax1.set_title("Resolution reaches a plateau before token cost does")
    ax1.grid(alpha=0.2)
    ax1.legend(first + second, [line.get_label() for line in first + second], frameon=False, loc="center right")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    plot(args.plot)
    print(run_sweep())
