"""Integrated public API and release harness for Chapter 22.

Trace grading and task-cluster statistics live in focused sibling modules;
this facade keeps one executable evaluation and release surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluation_statistics import (
    paired_cluster_uncertainty,
    pass_at_k_estimate,
    pass_pow_k_estimate,
    percentile,
    slice_rates,
    task_metrics,
)
from trace_grading import (
    DEFAULT_JUDGE,
    DEFAULT_TRACES,
    ROOT,
    cohen_kappa,
    grade_traces,
    judge_report,
    load_jsonl,
)


def release_report(
    traces: Path = DEFAULT_TRACES,
    calibration: Path = DEFAULT_JUDGE,
    *,
    margin: float = 0.02,
    slice_floor: float = 0.75,
) -> dict[str, Any]:
    """Build the evidence packet and apply its predeclared release rule."""
    rows = grade_traces(traces)
    judge = judge_report(calibration)
    baseline = task_metrics(rows, "baseline")
    candidate = task_metrics(rows, "candidate")
    uncertainty = paired_cluster_uncertainty(baseline, candidate)
    base_slices = slice_rates(rows, "baseline")
    cand_slices = slice_rates(rows, "candidate")
    golden = sorted(
        task_id
        for task_id in baseline
        if any(row["task_id"] == task_id and row["golden"] for row in rows)
        and candidate[task_id]["pass_rate"] < baseline[task_id]["pass_rate"]
    )
    reasons: list[str] = []
    if judge["kappa"] < 0.60 or judge["fail_recall"] < 0.70 or judge["position_flip_rate"] > 0.15:
        reasons.append("judge calibration does not meet the contract")
    if uncertainty["low"] < -margin:
        reasons.append("paired lower bound exceeds the allowed quality loss")
    weak_slices = [name for name, rate in cand_slices.items() if rate < slice_floor]
    if weak_slices:
        reasons.append("candidate misses slice floor: " + ", ".join(weak_slices))
    if golden:
        reasons.append("must-not-break regression: " + ", ".join(golden))
    return {
        "verdict": "SHIP" if not reasons else "BLOCK",
        "reasons": reasons,
        "judge": judge,
        "baseline": baseline,
        "candidate": candidate,
        "uncertainty": uncertainty,
        "slices": {"baseline": base_slices, "candidate": cand_slices},
        "trajectory_f1": {
            "baseline": fmean(value["trajectory_f1"] for value in baseline.values()),
            "candidate": fmean(value["trajectory_f1"] for value in candidate.values()),
        },
    }


def plot_reliability():
    """Plot analytic capability and reliability curves."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-22"
    ks = list(range(1, 9))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for p, marker in ((0.60, "o"), (0.80, "s"), (0.95, "^")):
        axes[0].plot(ks, [1 - (1 - p) ** k for k in ks], marker=marker, label=f"p={p:.2f}")
        axes[1].plot(ks, [p**k for k in ks], marker=marker, label=f"p={p:.2f}")
    axes[0].set_title("pass@k: at least one succeeds")
    axes[1].set_title("pass$^k$: every run succeeds")
    for axis in axes:
        axis.set(xlabel="k repeated runs", ylabel="probability", ylim=(0, 1.02))
        axis.grid(alpha=0.25)
    axes[1].legend(title="single-run rate")
    fig.tight_layout()
    return fig


def plot_release(report: dict[str, Any]):
    """Plot paired task deltas and predeclared workload slices."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-22"
    deltas = report["uncertainty"]["task_deltas"]
    slices = report["slices"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].bar(range(len(deltas)), list(deltas.values()), color="0.65", edgecolor="black")
    axes[0].set(title="Candidate minus baseline by task", xlabel="paired task cluster", ylabel="pass-rate delta")
    axes[0].set_xticks(range(len(deltas)), [str(i + 1) for i in range(len(deltas))])
    names = list(slices["baseline"])
    x = list(range(len(names)))
    axes[1].bar([i - 0.18 for i in x], [slices["baseline"][n] for n in names], 0.36, label="baseline", color="white", edgecolor="black", hatch="//")
    axes[1].bar([i + 0.18 for i in x], [slices["candidate"][n] for n in names], 0.36, label="candidate", color="0.55", edgecolor="black")
    axes[1].set(title="Predeclared language slices", ylabel="trial pass rate", ylim=(0, 1.02))
    axes[1].set_xticks(x, names, rotation=15)
    axes[1].legend()
    fig.tight_layout()
    return fig


def main() -> int:
    """Run the deterministic build and optionally save its plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = release_report(args.traces, args.calibration)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability().savefig(args.output_dir / "reliability.svg", format="svg", metadata={"Date": None})
        plot_release(report).savefig(args.output_dir / "release-evidence.svg", format="svg", metadata={"Date": None})
    return 0 if report["verdict"] == "SHIP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
