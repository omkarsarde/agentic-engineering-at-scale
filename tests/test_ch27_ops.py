"""Focused operational invariants for Chapter 27."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch27"
sys.path.insert(0, str(CODE))

from ops_console import (  # noqa: E402
    ApprovalRecord,
    RunRecord,
    autonomy_action,
    burn_rate,
    compute_slis,
    drift_by_tenant,
    emit_linked_run,
    hitl_metrics,
    run_fixture,
    sanitize_attributes,
)


def test_approval_pause_uses_two_linked_traces() -> None:
    spans, first_trace = emit_linked_run()
    assert len({span.context.trace_id for span in spans}) == 2
    assert sum(len(span.links) for span in spans) == 1
    assert any(f"{link.context.trace_id:032x}" == first_trace for span in spans for link in span.links)


def test_trace_cost_is_diagnostic_not_missing() -> None:
    report = run_fixture()
    assert report["diagnostic_trace_cost"] == 0.013
    assert report["span_count"] == 5


def test_per_tenant_drift_exposes_hidden_slice() -> None:
    baseline = [
        RunRecord("b1", "a", True, True, 1, 10, 1, 0.9),
        RunRecord("b2", "b", True, True, 1, 10, 1, 0.9),
    ]
    current = [
        RunRecord("c1", "a", True, True, 1, 10, 1, 1.0),
        RunRecord("c2", "b", False, False, 0, 10, 1, 0.4),
    ]
    assert drift_by_tenant(baseline, current, 0.2) == {"b": 0.5}


def test_journey_slis_use_outcomes_not_http_status() -> None:
    records = [
        RunRecord("1", "a", True, True, 1, 100, 1.0, 1.0),
        RunRecord("2", "a", True, False, 2, 200, 2.0, 0.0),
    ]
    slis = compute_slis(records)
    assert slis["success_and_grounded_rate"] == 0.5
    assert slis["exactly_one_effect_rate"] == 0.5
    assert slis["cost_per_successful_task"] == 3.0


def test_burn_controls_autonomy_but_safety_is_not_budgeted() -> None:
    burn = burn_rate(9_655, 10_000, 0.995)
    assert round(burn, 1) == 6.9
    assert autonomy_action(burn) == "require_review_and_page"
    assert autonomy_action(0.1, safety_violation=True) == "stop_new_effects"


def test_hitl_metrics_flag_fast_near_unanimous_approval() -> None:
    records = [ApprovalRecord(str(i), "approve", 0.2) for i in range(99)]
    records.append(ApprovalRecord("99", "deny", 0.3))
    metrics = hitl_metrics(records)
    assert metrics["approval_rate"] == 0.99
    assert metrics["rubber_stamp_signal"] is True


def test_redaction_is_content_off_and_secret_on() -> None:
    attrs = {
        "authorization": "Bearer abc.secret",
        "app.tenant.id": "tenant-raw",
        "gen_ai.input.messages": "private prompt",
        "error.message": "request used Bearer xyz.123",
    }
    cleaned = sanitize_attributes(attrs)
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["app.tenant.id"] != "tenant-raw"
    assert cleaned["gen_ai.input.messages"] == "[CONTENT_DISABLED]"
    assert "xyz.123" not in cleaned["error.message"]
