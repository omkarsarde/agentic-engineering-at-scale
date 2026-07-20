"""Render Chapter 6 numeric evidence with print-safe encodings."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "chapter-06"


def render_figures(memory_rows, probe, optimizer_rows, cost_rows, out_dir: Path) -> None:
    """Write the memory, optimizer, and cost SVGs from saved numeric rows."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    components = ("parameters", "gradients", "master_weights", "optimizer_moments", "activations")
    hatches = ("//", "\\\\", "..", "xx", "")
    bottom = np.zeros(len(memory_rows))
    for component, hatch in zip(components, hatches):
        values = np.asarray([float(row[component]) / 2**20 for row in memory_rows])
        axes[0].bar(
            [row["strategy"] for row in memory_rows],
            values,
            bottom=bottom,
            label=component.replace("_", " "),
            hatch=hatch,
            edgecolor="#444444",
            linewidth=0.4,
        )
        bottom += values
    axes[0].set(ylabel="Predicted steady-state MiB per rank")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].legend(frameon=False, fontsize=6.5)
    labels = ["without checkpoint", "with checkpoint"]
    saved = probe["saved_tensor_bytes"]
    axes[1].bar(
        labels,
        [saved["without_checkpoint"] / 2**20, saved["with_checkpoint"] / 2**20],
        color=["#a14f3b", "#2a7f9e"],
        hatch=["//", ".."],
        edgecolor="#444444",
    )
    axes[1].set(ylabel="Autograd-saved tensor MiB", title="Checkpoint recomputation tradeoff")
    axes[1].tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(out_dir / "memory-checkpoint.svg", format="svg", metadata={"Date": None})
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for optimizer, linestyle in (("AdamW", "--"), ("Muon-like", "-")):
        selected = [row for row in optimizer_rows if row["optimizer"] == optimizer]
        axis.plot(
            [row["step"] for row in selected],
            [row["training_loss"] for row in selected],
            linestyle=linestyle,
            label=optimizer,
        )
    schedule = [row for row in optimizer_rows if row["optimizer"] == "AdamW"]
    twin = axis.twinx()
    twin.plot(
        [row["step"] for row in schedule],
        [row["schedule_multiplier"] for row in schedule],
        color="#777777",
        linestyle=":",
        label="WSD multiplier",
    )
    axis.set(xlabel="Optimizer step", ylabel="Training cross-entropy (nats/token)")
    twin.set(ylabel="Learning-rate multiplier", ylim=(0, 1.08))
    lines, labels = axis.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    axis.legend(lines + extra, labels + extra_labels, frameon=False)
    figure.tight_layout()
    figure.savefig(out_dir / "optimizer-curves.svg", format="svg", metadata={"Date": None})
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    mfus = [100 * float(row["mfu"]) for row in cost_rows]
    axes[0].plot(
        mfus, [float(row["training_cost"]) / 1e6 for row in cost_rows], marker="o"
    )
    axes[0].set(xlabel="MFU (%)", ylabel="Illustrative training cost (million USD)")
    axes[1].plot(
        mfus, [float(row["wall_days"]) for row in cost_rows], marker="s", linestyle="--"
    )
    axes[1].set(xlabel="MFU (%)", ylabel="Wall time (days)")
    figure.tight_layout()
    figure.savefig(out_dir / "cost-sensitivity.svg", format="svg", metadata={"Date": None})
    plt.close(figure)
