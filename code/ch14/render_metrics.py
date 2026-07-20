"""Render the measured Chapter 14 chunk sweep as deterministic SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

from chapter_build import run


def render(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "ch14-rag-chunk-sweep"
    rows = run()["chunk_sweep"]
    sizes = [row["chunk_size_words"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))

    axes[0].plot(sizes, [row["recall_at_5"] for row in rows], color="#245b78", marker="o", linestyle="-", linewidth=2, label="Retrieval recall@5")
    axes[0].plot(sizes, [row["faithfulness"] for row in rows], color="#2e6f57", marker="s", linestyle="--", linewidth=2, label="Citation support")
    axes[0].plot(sizes, [row["answer_similarity"] for row in rows], color="#9a5b2f", marker="^", linestyle=":", linewidth=2.2, label="Lexical answer F1")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score on deterministic fixture")
    axes[0].set_title("Stage-local quality")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    axes[1].plot(sizes, [row["chunks"] for row in rows], color="#4b5563", marker="D", linewidth=2)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Indexed chunks (records)")
    axes[1].set_title("Index footprint proxy")
    for axis in axes:
        axis.set_xlabel("Chunk size (whitespace words)")
        axis.set_xticks(sizes)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Chunking changed cost more than recall on this small corpus", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
