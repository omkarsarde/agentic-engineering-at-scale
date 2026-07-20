"""Run the Chapter 15 comparison and generate its measured figure."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from agentic_retrieval import EvidenceIndex, agentic, correct, one_shot
from fixture import corpus, questions


def evaluate() -> dict[str, object]:
    """Evaluate a fixed-context frontier and the iterative executor."""
    index = EvidenceIndex(corpus())
    evaluation_set = questions()
    rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    strategies = (
        ("one-shot k=1", lambda query: one_shot(index, query, k=1)),
        ("one-shot k=2", lambda query: one_shot(index, query, k=2)),
        ("one-shot k=3", lambda query: one_shot(index, query, k=3)),
        ("one-shot k=4", lambda query: one_shot(index, query, k=4)),
        ("agentic", lambda query: agentic(index, query)),
    )
    for name, strategy in strategies:
        results = []
        for query in evaluation_set:
            result = strategy(query)
            results.append(result)
            rows.append(
                {
                    "strategy": name,
                    "query_id": query.id,
                    "expected": query.expected,
                    "answer": result.answer,
                    "supported_correct": correct(result, query, index),
                    "stop": result.stop,
                    "search_calls": result.search_calls,
                    "candidate_documents": result.candidate_documents,
                    "evidence": [asdict(edge) for edge in result.evidence],
                    "rejected_ids": result.rejected_ids,
                }
            )
        summary[name] = {
            "supported_answer_accuracy": sum(
                correct(result, query, index)
                for result, query in zip(results, evaluation_set)
            )
            / len(results),
            "mean_search_calls": sum(result.search_calls for result in results) / len(results),
            "mean_verified_candidate_documents": sum(
                result.candidate_documents for result in results
            )
            / len(results),
            "mean_rejected_documents": sum(
                len(result.rejected_ids) for result in results
            )
            / len(results),
            "mean_evidence_edges": sum(len(result.evidence) for result in results)
            / len(results),
        }
    return {
        "experiment_contract": "executor-only; gold compiled plans exclude planner error and cost",
        "summary": summary,
        "rows": rows,
    }


def render(metrics: dict[str, object], output: Path) -> None:
    """Render a print-safe quality/context frontier from measured fixture data."""
    plt.rcParams["svg.hashsalt"] = "chapter-15"
    summary = metrics["summary"]
    assert isinstance(summary, dict)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fixed_names = (
        "one-shot k=1",
        "one-shot k=2",
        "one-shot k=3",
        "one-shot k=4",
    )
    fixed_x = [
        float(summary[name]["mean_verified_candidate_documents"])
        for name in fixed_names
    ]
    fixed_y = [
        100 * float(summary[name]["supported_answer_accuracy"])
        for name in fixed_names
    ]
    ax.plot(fixed_x, fixed_y, color="0.35", marker="o", linestyle="--", label="fixed context")
    for name, x, y in zip(fixed_names, fixed_x, fixed_y):
        calls = float(summary[name]["mean_search_calls"])
        ax.annotate(
            f"{name.removeprefix('one-shot ')} · {y:.0f}% · {calls:.1f} call",
            (x, y),
            xytext=(7, -14 if name in {"one-shot k=2", "one-shot k=3"} else 7),
            textcoords="offset points",
        )
    agentic_values = summary["agentic"]
    agentic_x = float(agentic_values["mean_verified_candidate_documents"])
    agentic_y = 100 * float(agentic_values["supported_answer_accuracy"])
    agentic_calls = float(agentic_values["mean_search_calls"])
    ax.plot(agentic_x, agentic_y, color="0.0", marker="s", linestyle="none", markersize=8)
    ax.annotate(
        f"iterative · {agentic_y:.0f}% · {agentic_calls:.1f} calls",
        (agentic_x, agentic_y),
        xytext=(7, -14),
        textcoords="offset points",
    )
    x_max = max(*fixed_x, agentic_x)
    ax.set(
        xlabel="Mean verified candidate documents inspected (executor only)",
        ylabel="Supported-answer accuracy (%)",
        xlim=(0, x_max * 1.18),
        ylim=(0, 105),
    )
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate()
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    render(metrics, args.figure)


if __name__ == "__main__":
    main()
