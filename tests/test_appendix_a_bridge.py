"""Focused decision and distributed-systems bridge tests."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "appa"
sys.path.insert(0, str(CODE))

from idempotent_worker import build_report, fixture_deliveries  # noqa: E402
from render_pareto import pareto_names  # noqa: E402
from voi_triage import value_of_signal  # noqa: E402


def test_duplicate_fixture_is_at_least_once() -> None:
    deliveries = fixture_deliveries()
    assert len(deliveries) == 24
    assert len({task.key for task in deliveries}) == 20


def test_local_ledger_collapses_queue_duplicates() -> None:
    report = build_report()
    assert report["naive"]["effects"] == 24
    assert report["local_ledger"]["effects"] == 20
    assert report["local_ledger"]["states"] == {"committed": 20}


def test_effect_before_commit_remains_ambiguous_locally() -> None:
    report = build_report()
    assert report["ambiguous_window"]["crashes"] == 1
    assert report["ambiguous_window"]["effects"] == 21
    assert report["ambiguous_window"]["states"] == {"committed": 20}


def test_provider_idempotency_closes_fixture_effect_duplication() -> None:
    report = build_report()
    assert report["provider_key"]["calls"] == 21
    assert report["provider_key"]["effects"] == 20


def test_information_has_positive_value_before_observation_cost() -> None:
    value = value_of_signal(0.2, 0.85, 0.9, 10.0, 1.0)
    assert round(value, 3) == 0.42


def test_pareto_frontier_removes_dominated_configuration() -> None:
    frontier = pareto_names()
    assert "medium/direct" not in frontier
    assert "small/direct" in frontier
    assert "large/ensemble" in frontier
