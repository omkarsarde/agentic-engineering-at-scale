"""Focused evidence-led complexity and failure-game tests."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch32"
sys.path.insert(0, str(CODE))

from readiness_ledger import (  # noqa: E402
    RungReport,
    ablation_delta,
    attack_surface_edges,
    earn_decisions,
    failure_summary,
    fixture_injections,
    fixture_ladder,
)


def test_pass_pow_k_penalizes_inconsistent_success() -> None:
    report = fixture_ladder()[4]
    assert report.pass_pow_k(4) < report.task_success
    assert round(report.pass_pow_k(4), 4) == 0.8145


def test_invalid_measurement_is_rejected() -> None:
    try:
        RungReport(0, "bad", 10, 11, 1, 0.0, 0.0, 0, 0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("successes above trials were accepted")


def test_attack_surface_is_enumerated_not_vaguely_scored() -> None:
    edges = attack_surface_edges(["retrieval", "adaptive_loop", "memory", "write_tool"])
    assert "untrusted_content_to_model" in edges
    assert "untrusted_content_to_persistent_state" in edges
    assert "model_to_external_effect" in edges
    assert len(edges) == len(set(edges))


def test_adaptive_loop_and_memory_do_not_earn_fixture_cost() -> None:
    decisions = {row["rung"]: row["decision"] for row in earn_decisions(fixture_ladder())}
    assert decisions[5] == "cut_or_ablate"
    assert decisions[6] == "cut_or_ablate"
    assert decisions[7] == "earned"


def test_ablation_reports_all_seven_axis_consequences() -> None:
    reports = fixture_ladder()
    delta = ablation_delta(reports[6], reports[4])
    assert delta["success_delta"] == 0.0
    assert delta["cost_saved_usd"] > 0
    assert delta["latency_saved_ms"] > 0
    assert delta["attack_edges_removed"] > 0
    assert delta["operator_burden_removed"] > 0


def test_every_injection_is_contained_but_one_needs_postmortem() -> None:
    summary = failure_summary(fixture_injections())
    assert summary["injections"] == 10
    assert summary["contained"] == 10
    assert summary["detected"] == 9
    assert summary["recovered"] == 9
    assert summary["needs_postmortem"] == ["half_applied_migration"]


def test_each_failure_becomes_a_named_regression_test() -> None:
    summary = failure_summary(fixture_injections())
    assert len(summary["regression_tests"]) == 10
    assert all(name.startswith("test_") for name in summary["regression_tests"])
