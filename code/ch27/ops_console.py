"""Deterministic traces, SLIs, drift, HITL metrics, and redaction for Chapter 27."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Iterable

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    tenant: str
    success: bool
    grounded: bool
    effect_count: int
    ttft_ms: float
    cost: float
    score: float


@dataclass(frozen=True)
class ApprovalRecord:
    action_id: str
    decision: str  # approve | deny | override | abandon
    latency_s: float


def build_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Create an isolated in-memory OTel pipeline for examples and tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("agentic-book-ch27"), exporter


def emit_linked_run() -> tuple[list[Any], str]:
    """Represent a durable approval pause as two linked, bounded traces."""
    tracer, exporter = build_tracer()
    with tracer.start_as_current_span(
        "invoke_agent",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "app.run.id": "run-17",
            "app.release.hash": "bundle-42",
        },
    ) as first:
        with tracer.start_as_current_span(
            "chat",
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "fixture-model",
                "gen_ai.usage.input_tokens": 20_000,
                "gen_ai.usage.output_tokens": 2_000,
                "app.cost": 0.013,
            },
        ):
            pass
        with tracer.start_as_current_span(
            "request_approval",
            attributes={"app.action.id": "refund:A-17:v1"},
        ):
            pass
        first_context = first.get_span_context()

    with tracer.start_as_current_span(
        "resume_agent",
        links=[Link(first_context, attributes={"app.link.reason": "approval_resume"})],
        attributes={"app.run.id": "run-17", "app.approval.decision": "approve"},
    ):
        with tracer.start_as_current_span(
            "execute_tool",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "app.action.id": "refund:A-17:v1",
                "app.effect.receipt": "receipt-1",
            },
        ):
            pass

    spans = list(exporter.get_finished_spans())
    return spans, f"{first_context.trace_id:032x}"


def trace_cost(spans: Iterable[Any]) -> float:
    """Read diagnostic cost from spans; billing still belongs to the usage ledger."""
    return sum(float(span.attributes.get("app.cost", 0.0)) for span in spans)


def sampled(run_id: str, sample_rate: float) -> bool:
    """Choose a stable sample without coupling tests to random state."""
    if not 0 <= sample_rate <= 1:
        raise ValueError("sample rate must be between zero and one")
    bucket = int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < sample_rate


def score_sample(records: Iterable[RunRecord], sample_rate: float) -> list[RunRecord]:
    """Return records selected for an asynchronous deterministic scorer."""
    return [record for record in records if sampled(record.run_id, sample_rate)]


def drift_by_tenant(
    baseline: Iterable[RunRecord], current: Iterable[RunRecord], threshold: float
) -> dict[str, float]:
    """Return per-tenant score drops that exceed a declared threshold."""
    base: dict[str, list[float]] = {}
    live: dict[str, list[float]] = {}
    for record in baseline:
        base.setdefault(record.tenant, []).append(record.score)
    for record in current:
        live.setdefault(record.tenant, []).append(record.score)
    alerts: dict[str, float] = {}
    for tenant in sorted(base.keys() & live.keys()):
        drop = mean(base[tenant]) - mean(live[tenant])
        if drop >= threshold:
            alerts[tenant] = round(drop, 6)
    return alerts


def nearest_rank(values: list[float], percentile: float) -> float:
    """Compute a deterministic nearest-rank percentile."""
    if not values:
        raise ValueError("values cannot be empty")
    rank = max(1, math.ceil(percentile * len(values)))
    return sorted(values)[rank - 1]


def compute_slis(records: list[RunRecord]) -> dict[str, float]:
    """Compute journey-level quality, effect, latency, and cost indicators."""
    if not records:
        raise ValueError("records cannot be empty")
    successful = [record for record in records if record.success and record.grounded]
    return {
        "success_and_grounded_rate": len(successful) / len(records),
        "exactly_one_effect_rate": sum(r.effect_count == 1 for r in records) / len(records),
        "p95_ttft_ms": nearest_rank([r.ttft_ms for r in records], 0.95),
        "cost_per_successful_task": sum(r.cost for r in records) / len(successful),
    }


def burn_rate(good: int, total: int, slo_target: float) -> float:
    """Compare observed bad-event rate with the SLO's allowed bad rate."""
    if not 0 < slo_target < 1 or not 0 <= good <= total or total <= 0:
        raise ValueError("invalid SLO counts or target")
    observed_bad = 1 - good / total
    allowed_bad = 1 - slo_target
    return observed_bad / allowed_bad


def days_to_exhaustion(window_days: float, remaining_fraction: float, burn: float) -> float:
    """Estimate time to exhaust a budget under a constant burn rate."""
    if burn <= 0:
        return math.inf
    return window_days * remaining_fraction / burn


def autonomy_action(burn: float, safety_violation: bool = False) -> str:
    """Translate fleet evidence into a bounded runtime posture."""
    if safety_violation:
        return "stop_new_effects"
    if burn >= 14.4:
        return "read_only_and_freeze_rollout"
    if burn >= 6.0:
        return "require_review_and_page"
    if burn >= 1.0:
        return "open_ticket"
    return "normal"


def hitl_metrics(records: list[ApprovalRecord]) -> dict[str, float | bool]:
    """Measure whether a human queue operates or merely rubber-stamps."""
    if not records:
        raise ValueError("approval records cannot be empty")
    completed = [r for r in records if r.decision != "abandon"]
    approvals = [r for r in completed if r.decision == "approve"]
    overrides = [r for r in completed if r.decision == "override"]
    approval_rate = len(approvals) / len(completed) if completed else 0.0
    median_latency = sorted(r.latency_s for r in completed)[len(completed) // 2]
    return {
        "approval_rate": approval_rate,
        "override_rate": len(overrides) / len(completed) if completed else 0.0,
        "abandonment_rate": (len(records) - len(completed)) / len(records),
        "median_decision_s": median_latency,
        "rubber_stamp_signal": approval_rate >= 0.98 and median_latency < 1.0,
    }


SECRET_KEY = re.compile(r"authorization|api[_-]?key|token|secret", re.IGNORECASE)
BEARER = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")


def sanitize_attributes(
    attributes: dict[str, Any], capture_content: bool = False
) -> dict[str, Any]:
    """Redact secrets and minimize identity/content before span creation."""
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if SECRET_KEY.search(key):
            cleaned[key] = "[REDACTED]"
        elif key in {"gen_ai.input.messages", "gen_ai.output.messages"} and not capture_content:
            cleaned[key] = "[CONTENT_DISABLED]"
        elif key == "app.tenant.id":
            cleaned[key] = hashlib.sha256(str(value).encode()).hexdigest()[:12]
        elif isinstance(value, str):
            cleaned[key] = BEARER.sub("Bearer [REDACTED]", value)
        else:
            cleaned[key] = value
    return cleaned


def fixture_records() -> list[RunRecord]:
    return [
        RunRecord("r1", "alpha", True, True, 1, 80, 0.010, 0.96),
        RunRecord("r2", "alpha", True, True, 1, 90, 0.012, 0.94),
        RunRecord("r3", "beta", True, False, 1, 110, 0.014, 0.45),
        RunRecord("r4", "beta", False, False, 0, 400, 0.020, 0.30),
    ]


def run_fixture() -> dict[str, Any]:
    spans, first_trace = emit_linked_run()
    burn = burn_rate(9_655, 10_000, 0.995)
    records = fixture_records()
    approvals = [
        ApprovalRecord(f"a{i}", "approve", 0.2) for i in range(99)
    ] + [ApprovalRecord("a99", "deny", 0.3)]
    return {
        "span_count": len(spans),
        "trace_count": len({span.context.trace_id for span in spans}),
        "linked_resume": sum(len(span.links) for span in spans),
        "first_trace": first_trace,
        "diagnostic_trace_cost": trace_cost(spans),
        "slis": compute_slis(records),
        "burn_rate": round(burn, 1),
        "days_to_exhaustion": round(days_to_exhaustion(30, 0.95, burn), 1),
        "runtime_action": autonomy_action(burn),
        "hitl": hitl_metrics(approvals),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_fixture(), indent=2))
