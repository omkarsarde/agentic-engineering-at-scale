"""Trace isolation, deterministic grading, and judge calibration."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACES = ROOT / "fixtures" / "traces.jsonl"
DEFAULT_JUDGE = ROOT / "fixtures" / "judge_calibration.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a JSON Lines fixture."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def grade_traces(path: Path = DEFAULT_TRACES) -> list[dict[str, Any]]:
    """Expand task fixtures into graded, isolated trial records."""
    graded: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    environments: set[str] = set()
    for task in load_jsonl(path):
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        expected_tools = list(task["expected_tools"])
        for system in ("baseline", "candidate"):
            runs = task["runs"][system]
            if len(runs) < 2:
                raise ValueError(f"{task_id}/{system} needs repeated trials")
            for trial, run in enumerate(runs):
                environment_id = f"{task['snapshot']}:{system}:{trial}"
                if environment_id in environments:
                    raise ValueError(f"environment reused: {environment_id}")
                environments.add(environment_id)
                actual_tools = list(run["tools"])
                overlap = sum((Counter(actual_tools) & Counter(expected_tools)).values())
                precision = overlap / len(actual_tools) if actual_tools else 0.0
                recall = overlap / len(expected_tools) if expected_tools else 1.0
                trajectory_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                state_pass = run["final_state"] == task["expected_state"]
                schema_pass = bool(run["schema_valid"])
                policy_pass = bool(run["policy_ok"])
                judge_pass = run["judge_label"] == "PASS"
                graded.append(
                    {
                        "task_id": task_id,
                        "slice": task["slice"],
                        "golden": bool(task["golden"]),
                        "system": system,
                        "trial": trial,
                        "environment_id": environment_id,
                        "state_pass": state_pass,
                        "schema_pass": schema_pass,
                        "policy_pass": policy_pass,
                        "judge_pass": judge_pass,
                        "success": state_pass and schema_pass and policy_pass and judge_pass,
                        "trajectory_f1": trajectory_f1,
                        "exact_path": actual_tools == expected_tools,
                        "looped": any(a == b for a, b in zip(actual_tools, actual_tools[1:])),
                    }
                )
    return graded


def cohen_kappa(first: list[str], second: list[str]) -> float:
    """Compute chance-corrected agreement for two categorical raters."""
    if not first or len(first) != len(second):
        raise ValueError("raters need equal, non-empty label lists")
    labels = set(first) | set(second)
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    ca, cb = Counter(first), Counter(second)
    expected = sum((ca[label] / len(first)) * (cb[label] / len(first)) for label in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)


def judge_report(path: Path = DEFAULT_JUDGE) -> dict[str, Any]:
    """Summarize pointwise calibration and pairwise position consistency."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    human: list[str] = []
    judge: list[str] = []
    by_slice: dict[str, list[bool]] = defaultdict(list)
    for cell in payload["pointwise_cells"]:
        count = int(cell["count"])
        human.extend([cell["human"]] * count)
        judge.extend([cell["judge"]] * count)
        by_slice[cell["slice"]].extend([cell["human"] == cell["judge"]] * count)
    fail_total = sum(label == "FAIL" for label in human)
    fail_caught = sum(a == b == "FAIL" for a, b in zip(human, judge))
    pair_total = sum(int(cell["count"]) for cell in payload["pairwise_cells"])
    pair_flips = sum(int(cell["count"]) for cell in payload["pairwise_cells"] if cell["ab"] != cell["ba_normalized"])
    return {
        "n": len(human),
        "agreement": fmean(a == b for a, b in zip(human, judge)),
        "kappa": cohen_kappa(human, judge),
        "fail_recall": fail_caught / fail_total,
        "position_flip_rate": pair_flips / pair_total,
        "slice_agreement": {name: fmean(values) for name, values in sorted(by_slice.items())},
    }
