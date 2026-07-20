"""Run the deterministic Chapter 4 architecture-economics build."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

from config_reader import ArchitectureEstimate, estimate_config
from moe_min import mqar_capacity_curve, train_router


matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_router_figure(router_rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    colors = {"unbalanced": "#a14f3b", "balanced": "#2a7f9e"}
    for axis, condition in zip(axes, ("unbalanced", "balanced")):
        selected = [row for row in router_rows if row["condition"] == condition]
        axis.bar(
            [int(row["expert"]) for row in selected],
            [float(row["token_fraction"]) for row in selected],
            color=colors[condition],
        )
        axis.axhline(0.25, color="#555555", linestyle="--", linewidth=1)
        axis.set(
            title=condition.capitalize(),
            xlabel="Expert",
            xticks=range(4),
            ylim=(0, 1.05),
        )
    axes[0].set_ylabel("Fraction of routed tokens")
    fig.tight_layout()
    fig.savefig(out_dir / "router-load.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def _save_mqar_figure(mqar_rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.1))
    styles = {
        "full attention": ("#2a7f9e", "o"),
        "fixed state (8 slots)": ("#a14f3b", "s"),
        "fixed state (32 slots)": ("#7356a8", "^"),
    }
    for memory, (color, marker) in styles.items():
        selected = [row for row in mqar_rows if row["memory"] == memory]
        axis.plot(
            [int(row["pairs"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            color=color,
            marker=marker,
            label=memory,
        )
    axis.set_xscale("log", base=2)
    axis.set(
        xlabel="Key-value pairs written before the query",
        ylabel="Exact recall rate",
        ylim=(0, 1.04),
    )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fixed-state-mqar.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def _save_landscape_figure(landscape_rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    xs = [float(row["total_b"]) for row in landscape_rows]
    ys = [float(row["active_b"]) for row in landscape_rows]
    axis.scatter(xs, ys, s=48, color="#2a7f9e", zorder=3)
    offsets = {
        "DeepSeek-V4-Pro": (8, 7),
        "DeepSeek-V4-Flash": (8, -19),
        "Kimi K2.5": (8, 7),
        "Qwen3.5 397B-A17B": (-96, -22),
        "GLM-5": (8, 7),
        "MiniMax-M2": (8, -20),
        "gpt-oss-120b": (8, 7),
        "Llama 4 Maverick": (10, 10),
    }
    for row in landscape_rows:
        offset = offsets[str(row["name"])]
        axis.annotate(
            str(row["name"]),
            (float(row["total_b"]), float(row["active_b"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if offset[0] < 0 else "left",
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5}
            if abs(offset[0]) > 20 or abs(offset[1]) > 15
            else None,
        )
    low, high = min(xs) * 0.85, max(xs) * 1.15
    axis.plot([low, high], [0.03 * low, 0.03 * high], "--", color="#7356a8", linewidth=1, label="3% active")
    axis.plot([low, high], [0.05 * low, 0.05 * high], ":", color="#a14f3b", linewidth=1.2, label="5% active")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        xlabel="Published total parameters (billions)",
        ylabel="Published active parameters per token (billions)",
    )
    axis.legend(frameon=False, loc="lower right")
    axis.grid(which="major", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "landscape-total-active.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def run_build(out_dir: Path) -> dict[str, object]:
    """Generate all Chapter 4 measurements from local, dated fixtures."""

    out_dir.mkdir(parents=True, exist_ok=True)
    estimates: list[ArchitectureEstimate] = [
        estimate_config(path)
        for path in sorted((HERE / "fixtures").glob("*/config.json"))
    ]
    estimate_rows = [estimate.as_dict() for estimate in estimates]

    router_runs = {
        "unbalanced": train_router(balance_weight=0.0),
        "balanced": train_router(balance_weight=0.5),
    }
    router_rows: list[dict[str, object]] = []
    for condition, result in router_runs.items():
        for expert, fraction in enumerate(result["load"]):
            router_rows.append(
                {
                    "condition": condition,
                    "expert": expert,
                    "token_fraction": fraction,
                    "accuracy": result["accuracy"],
                    "balance_loss": result["balance_loss"],
                    "z_loss": result["z_loss"],
                }
            )

    mqar_rows = mqar_capacity_curve()
    landscape = json.loads((HERE / "fixtures" / "landscape_models.json").read_text(encoding="utf-8"))
    landscape_rows = [
        {**row, "active_fraction": row["active_b"] / row["total_b"], "verified_on": landscape["verified_on"]}
        for row in landscape["models"]
    ]

    metrics: dict[str, object] = {
        "config_estimates": estimate_rows,
        "router_runs": router_runs,
        "mqar_capacity": mqar_rows,
        "landscape_models": landscape_rows,
        "experiment_contracts": {
            "router": {
                "seed": 5,
                "training_steps": 300,
                "tokens_per_step": 256,
                "claim": "seeded toy classification MoE; it demonstrates routing collapse, not model quality",
            },
            "mqar": {
                "trials_per_point": 4096,
                "claim": "synthetic hashed fixed-state capacity probe, not a language-model benchmark",
            },
            "parameter_parser": {
                "claim": "architecture estimate from selected config fields; conventions differ from publisher counts"
            },
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_csv(out_dir / "config-estimates.csv", estimate_rows)
    _write_csv(out_dir / "router-loads.csv", router_rows)
    _write_csv(out_dir / "mqar-capacity.csv", mqar_rows)
    _write_csv(out_dir / "landscape-models.csv", landscape_rows)

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-04"
    _save_router_figure(router_rows, out_dir)
    _save_mqar_figure(mqar_rows, out_dir)
    _save_landscape_figure(landscape_rows, out_dir)
    return metrics


if __name__ == "__main__":
    run_build(HERE / "generated")
