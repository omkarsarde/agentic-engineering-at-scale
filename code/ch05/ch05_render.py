"""Render Chapter 5 numeric evidence with print-safe encodings."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "chapter-05"


def save_scaling_figure(observations, law, extrapolation, out_dir: Path) -> None:
    """Render fit agreement and the fixed-compute optimum."""

    import matplotlib.pyplot as plt

    actual = np.asarray([row.loss for row in observations])
    fitted = np.asarray([float(law.loss(row.parameters, row.tokens)) for row in observations])
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].scatter(actual, fitted, color="#2a7f9e")
    limits = (min(actual.min(), fitted.min()) - 0.05, max(actual.max(), fitted.max()) + 0.05)
    axes[0].plot(limits, limits, "--", color="#555555", linewidth=1)
    axes[0].set(xlabel="Synthetic observed loss", ylabel="Fitted loss", xlim=limits, ylim=limits)

    compute = extrapolation["target_compute_flops"]
    optimum = extrapolation["optimal_parameters"]
    parameters = np.geomspace(optimum / 8, optimum * 8, 120)
    tokens = compute / (6 * parameters)
    profile = law.loss(parameters, tokens)
    axes[1].plot(parameters / 1e9, profile, color="#7356a8")
    axes[1].errorbar(
        [optimum / 1e9],
        [extrapolation["predicted_loss"]],
        yerr=[
            [max(0.0, extrapolation["predicted_loss"] - extrapolation["prediction_p05"])],
            [max(0.0, extrapolation["prediction_p95"] - extrapolation["predicted_loss"])],
        ],
        fmt="o",
        color="#a14f3b",
        capsize=3,
        label="fitted optimum, 90% residual-bootstrap predictive interval",
    )
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Parameters at fixed 10× compute (billions)", ylabel="Predicted loss")
    axes[1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(out_dir / "scaling-law-fit.svg", format="svg", metadata={"Date": None})
    plt.close(figure)


def save_pipeline_figure(stage_rows, cluster_rows, out_dir: Path) -> None:
    """Render gate survival and cluster-size evidence."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].bar(
        [str(row["stage"]) for row in stage_rows],
        [int(row["documents"]) for row in stage_rows],
        color="#2a7f9e",
    )
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].set(ylabel="Documents", title="Documents surviving each gate")
    cluster_sizes = Counter((row["cluster_id"], row["cluster_size"]) for row in cluster_rows)
    histogram = Counter(size for _, size in cluster_sizes)
    axes[1].bar(list(histogram), list(histogram.values()), color="#a14f3b")
    axes[1].set(xlabel="Documents in near-duplicate cluster", ylabel="Clusters")
    figure.tight_layout()
    figure.savefig(out_dir / "data-pipeline.svg", format="svg", metadata={"Date": None})
    plt.close(figure)


def save_training_figure(curves, fertility, summary, out_dir: Path) -> None:
    """Render training trajectories and tokenizer fertility."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    styles = {"raw": ("#a14f3b", "--"), "cleaned": ("#2a7f9e", "-")}
    for condition, (color, linestyle) in styles.items():
        selected = [row for row in curves if row["condition"] == condition]
        steps = sorted({int(row["step"]) for row in selected})
        values = [
            [float(row["training_loss"]) for row in selected if int(row["step"]) == step]
            for step in steps
        ]
        axes[0].plot(
            steps,
            [float(np.mean(row)) for row in values],
            color=color,
            linestyle=linestyle,
            label=f"{condition} train mean",
        )
        if len(summary["paired_seeds"]) > 1:
            axes[0].fill_between(
                steps,
                [min(row) for row in values],
                [max(row) for row in values],
                color=color,
                alpha=0.12,
            )
        axes[0].axhline(
            summary[condition]["heldout_loss"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{condition} held-out mean",
        )
    axes[0].set(xlabel="Optimizer step", ylabel="Cross-entropy (nats/token)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(
        [str(row["language"]) for row in fertility],
        [float(row["tokens_per_character"]) for row in fertility],
        color=["#2a7f9e", "#7356a8", "#a14f3b", "#4f7d4a", "#777777"],
    )
    axes[1].set(ylabel="Tokenizer tokens per Unicode character", xlabel="Language sample")
    figure.tight_layout()
    figure.savefig(out_dir / "data-quality-training.svg", format="svg", metadata={"Date": None})
    plt.close(figure)
