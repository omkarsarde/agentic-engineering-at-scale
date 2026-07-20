"""Render the Chapter 11 numeric figure from the deterministic lab."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from customization_lab import run_lab


def main() -> None:
    metrics = run_lab()
    scores = metrics["scores"]
    names = ["train", "dev", "task_test", "general_regression"]
    labels = ["Train", "Dev", "Untouched\ntask test", "General\nregression"]
    versions = [("Base NF4", "base_nf4"), ("QLoRA SFT", "qlora_sft"), ("+ logit KD", "logit_kd")]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "chapter-11-customization",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)

    x = np.arange(len(names))
    width = 0.24
    colors = ["#d9d9d9", "#8f8f8f", "#202020"]
    for index, ((label, key), color) in enumerate(zip(versions, colors, strict=True)):
        values = [scores[key][name] for name in names]
        bars = axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="#111111",
        )
        axes[0].bar_label(
            bars,
            labels=[f"{value:.0%}" for value in values],
            padding=2,
            fontsize=7,
        )
    regression_floor = scores["base_nf4"]["general_regression"] - 0.08
    axes[0].plot([2.6, 3.4], [regression_floor, regression_floor], color="#555555", linestyle=":", linewidth=1)
    axes[0].text(
        3.36,
        regression_floor + 0.006,
        "regression floor",
        ha="right",
        color="#444444",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )
    axes[0].set(ylim=(0, 1.08), ylabel="Accuracy", title="Every update faces the same four-set gate")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3)

    for method, marker, color in (("linear", "o", "#777777"), ("ties", "s", "#111111")):
        rows = [row for row in metrics["merge_sweep"] if row["method"] == method]
        axes[1].plot(
            [row["weight_a"] for row in rows],
            [row["task_mean"] for row in rows],
            marker=marker,
            color=color,
            label=method.upper() if method == "ties" else "Linear",
        )
    axes[1].axvline(0.5, color="#555555", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Adapter A weight (adapter B gets 1 − weight)",
        ylabel="Mean accuracy across tasks A and B",
        ylim=(0, 1.0),
        title="A merge method still needs a coefficient sweep",
    )
    axes[1].legend(frameon=False, loc="lower right")

    destination = Path(__file__).with_name("customization_metrics.svg")
    fig.savefig(destination, format="svg", metadata={"Date": None})
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
