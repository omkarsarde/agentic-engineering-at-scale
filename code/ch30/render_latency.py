"""Render the Chapter 30 first-audio latency comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from voice_loop import latency_fixture, latency_report


def plot(path: Path) -> None:
    """Write a directly labeled SVG from deterministic fixture data."""

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-30"
    rows = latency_fixture()
    labels = ["Turn end", "ASR", "Model", "TTS", "Network"]
    sequential = [
        int(sum(getattr(row, field) for row in rows) / len(rows))
        for field in ("turn_end", "asr", "model", "tts", "network")
    ]
    overlapped = [sequential[0], sequential[1] - 90, sequential[2], sequential[3] - 70, sequential[4]]
    colors = ["#315b8a", "#4c78a8", "#2f855a", "#72a98f", "#8a6d3b"]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 3.8))
    for row_index, values in enumerate((sequential, overlapped)):
        left = 0
        for label, value, color in zip(labels, values, colors, strict=True):
            axis.barh(row_index, value, left=left, color=color, label=label if row_index == 0 else None)
            if value >= 80:
                axis.text(left + value / 2, row_index, f"{value} ms", ha="center", va="center", color="white", fontsize=8)
            left += value
        axis.text(left - 8, row_index, f"{left} ms", ha="right", va="center", color="white", fontsize=9)
    axis.axvline(1000, color="#b13f3f", linestyle="--", linewidth=1.5)
    axis.text(1005, 1.45, "1 s budget", color="#8f3030", fontsize=9)
    axis.set_yticks([0, 1], ["Sequential", "Streaming overlap"])
    axis.set_xlabel("Mean first-audio critical path (milliseconds)")
    axis.set_title("Overlap helps; endpointing remains the largest latency term")
    axis.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    output_format = path.suffix.removeprefix(".") or "svg"
    fig.savefig(path, format=output_format, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    plot(args.plot)
    print(latency_report())
