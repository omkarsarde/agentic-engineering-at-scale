"""Focused tests for the Chapter 20 multi-agent runtime."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory


CODE = Path(__file__).parents[1] / "code" / "ch20"
sys.path.insert(0, str(CODE))
sys.modules.pop("fixture", None)

from fixture import CORPUS, run_fixture  # noqa: E402
from team_runtime import Orchestrator, SingleAgent, Workspace, should_split  # noqa: E402


def test_split_requires_all_three_conditions() -> None:
    assert should_split(parallelizable=True, isolation_boundary=True, measured_failure=True)
    assert not should_split(parallelizable=False, isolation_boundary=True, measured_failure=True)
    assert not should_split(parallelizable=True, isolation_boundary=False, measured_failure=True)
    assert not should_split(parallelizable=True, isolation_boundary=True, measured_failure=False)


def test_cost_matched_outcomes_are_equal() -> None:
    report = run_fixture()["cost_matched"]
    assert report["same_answer"] is True
    assert report["single"]["tokens"] == report["team"]["tokens"]
    assert report["single"]["score"] == report["team"]["score"] == 1.0


def test_parallel_team_reduces_critical_path() -> None:
    report = run_fixture()["cost_matched"]
    assert report["team_speedup"] == 2.0
    assert report["team"]["latency_units"] < report["single"]["latency_units"]


def test_missing_provenance_is_contained_and_attributed() -> None:
    fault = run_fixture()["fault_injection"]
    assert fault["status"] == "contained"
    assert fault["answer"] is None
    assert fault["failure"]["category"] == "inter-agent misalignment"
    assert fault["failure"]["agent"] == "worker-2"
    assert fault["failure"]["reason"] == "missing provenance"


def test_single_agent_fails_closed_below_budget() -> None:
    result = SingleAgent().run(CORPUS, token_budget=500)
    assert result.status == "budget_exhausted"
    assert result.answer is None


def test_worker_reports_are_immutable() -> None:
    with TemporaryDirectory(prefix="ch20-test-") as directory:
        workspace = Workspace(Path(directory))
        result = Orchestrator().run(CORPUS, workspace)
        assert result.status == "completed"
        assert len(tuple(Path(directory).glob("*.json"))) == 3
