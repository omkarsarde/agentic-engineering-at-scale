"""Run the Chapter 20 cost-matched team experiment and fault injection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from team_runtime import Orchestrator, SingleAgent, Workspace, result_dict


CORPUS = (
    {"shard_id": "americas", "finding": "Brazil", "evidence_id": "policy-7#L18"},
    {"shard_id": "emea", "finding": "France", "evidence_id": "policy-8#L11"},
    {"shard_id": "apac", "finding": "Japan", "evidence_id": "policy-9#L24"},
)


def run_fixture() -> dict[str, Any]:
    with TemporaryDirectory(prefix="ch20-team-") as directory:
        team = Orchestrator().run(CORPUS, Workspace(Path(directory) / "baseline"))
        single = SingleAgent().run(CORPUS, token_budget=team.tokens)
        faulted = Orchestrator().run(
            CORPUS,
            Workspace(Path(directory) / "faulted"),
            poison_worker="worker-2",
        )
    return {
        "cost_matched": {
            "single": result_dict(single),
            "team": result_dict(team),
            "team_speedup": round(single.latency_units / team.latency_units, 2),
            "same_answer": single.answer == team.answer,
        },
        "fault_injection": result_dict(faulted),
    }


def plot_report(report: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-20"
    path.parent.mkdir(parents=True, exist_ok=True)
    pair = report["cost_matched"]
    names = ["single", "team"]
    tokens = [pair[name]["tokens"] for name in names]
    latency = [pair[name]["latency_units"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    axes[0].bar(names, tokens, color=["#64748b", "#315b8a"])
    axes[0].set_title("Equal measured token budget")
    axes[0].set_ylabel("Token units")
    axes[1].bar(names, latency, color=["#64748b", "#2f855a"])
    axes[1].set_title("Parallelism buys critical-path time")
    axes[1].set_ylabel("Latency units")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    output = run_fixture()
    if args.plot:
        plot_report(output, args.plot)
    print(json.dumps(output, indent=2))
