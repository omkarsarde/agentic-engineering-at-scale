"""Evidence ledger for the Chapter 32 complexity ladder and failure game."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class RungReport:
    """One comparable measurement of an architecture rung.

    Args:
        rung: Zero-based rung number.
        capability: Capability added at this rung.
        trials: Number of independent golden-set attempts.
        successes: Attempts satisfying final-state and policy predicates.
        p95_latency_ms: End-to-end task latency at the 95th percentile.
        cost_per_task_usd: Unsampled variable cost per completed task.
        energy_joules_est: Explicitly estimated, not metered, energy per task.
        attack_edges: Count of enumerated high-risk trust/authority edges.
        approvals_per_100: Human approvals requested per 100 tasks.
        escalations_per_100: Human escalations requested per 100 tasks.

    Raises:
        ValueError: If counts, units, or probabilities are invalid.
    """

    rung: int
    capability: str
    trials: int
    successes: int
    p95_latency_ms: int
    cost_per_task_usd: float
    energy_joules_est: float
    attack_edges: int
    approvals_per_100: float
    escalations_per_100: float

    def __post_init__(self) -> None:
        if self.rung < 0 or self.trials <= 0 or not 0 <= self.successes <= self.trials:
            raise ValueError("invalid rung or trial counts")
        numeric = (self.p95_latency_ms, self.cost_per_task_usd, self.energy_joules_est)
        if any(value < 0 for value in numeric) or self.attack_edges < 0:
            raise ValueError("latency, cost, energy, and attack edges cannot be negative")
        if self.approvals_per_100 < 0 or self.escalations_per_100 < 0:
            raise ValueError("operator burden cannot be negative")

    @property
    def task_success(self) -> float:
        """Return the empirical task-and-policy success rate."""

        return self.successes / self.trials

    def pass_pow_k(self, repeats: int = 4) -> float:
        """Estimate pass^k: the probability all repeated attempts succeed."""

        if repeats <= 0:
            raise ValueError("repeats must be positive")
        return self.task_success**repeats

    @property
    def operator_burden(self) -> float:
        """Return approvals plus escalations per 100 tasks."""

        return self.approvals_per_100 + self.escalations_per_100


@dataclass(frozen=True)
class InjectionResult:
    """Observed survival result for one bounded failure injection."""

    name: str
    detected: bool
    contained: bool
    recovered: bool
    regression_test: str


def attack_surface_edges(capabilities: Iterable[str]) -> tuple[str, ...]:
    """Enumerate explicit high-risk edges introduced by capabilities."""

    mapping = {
        "retrieval": ("untrusted_content_to_model",),
        "read_tool": ("model_to_external_read",),
        "workflow": ("orchestrator_to_credentials",),
        "adaptive_loop": ("model_to_loop_budget", "tool_output_to_next_instruction"),
        "memory": ("untrusted_content_to_persistent_state", "memory_to_future_session"),
        "write_tool": ("model_to_external_effect",),
    }
    edges: list[str] = []
    for capability in capabilities:
        edges.extend(mapping.get(capability, ()))
    return tuple(dict.fromkeys(edges))


def fixture_ladder() -> list[RungReport]:
    """Return the deterministic support-agent ladder measurements."""

    rows = (
        (0, "model call", 68, 470, 0.0020, 5.8, 0, 0, 2),
        (1, "structured output", 77, 490, 0.0023, 6.1, 0, 0, 1),
        (2, "retrieval", 90, 690, 0.0062, 10.7, 1, 0, 3),
        (3, "one read-only tool", 91, 790, 0.0091, 13.5, 2, 0, 3),
        (4, "deterministic workflow", 95, 930, 0.0120, 17.4, 3, 3, 3),
        (5, "adaptive loop", 95, 1410, 0.0250, 35.8, 5, 7, 5),
        (6, "persistent memory", 95, 1510, 0.0300, 42.2, 7, 8, 6),
        (7, "human-gated write", 98, 1690, 0.0410, 50.9, 8, 28, 7),
    )
    return [
        RungReport(rung, name, 100, success, latency, cost, energy, attacks, approvals, escalations)
        for rung, name, success, latency, cost, energy, attacks, approvals, escalations in rows
    ]


def earn_decisions(reports: Iterable[RungReport], min_gain: float = 0.015) -> list[dict[str, object]]:
    """Recommend keeping only rungs that clear an evidence threshold."""

    materialized = list(reports)
    if not materialized:
        raise ValueError("reports cannot be empty")
    decisions: list[dict[str, object]] = []
    previous = materialized[0]
    decisions.append({"rung": previous.rung, "decision": "baseline", "success_gain": None})
    for report in materialized[1:]:
        gain = report.task_success - previous.task_success
        required_effect = report.capability == "human-gated write"
        earned = gain >= min_gain or required_effect
        decisions.append(
            {
                "rung": report.rung,
                "capability": report.capability,
                "decision": "earned" if earned else "cut_or_ablate",
                "success_gain": round(gain, 3),
                "cost_ratio": round(report.cost_per_task_usd / previous.cost_per_task_usd, 2),
                "attack_edges_added": report.attack_edges - previous.attack_edges,
                "operator_burden_added": report.operator_burden - previous.operator_burden,
            }
        )
        previous = report
    return decisions


def ablation_delta(full: RungReport, ablated: RungReport) -> dict[str, float]:
    """Measure the consequence of removing one capability layer."""

    return {
        "success_delta": round(full.task_success - ablated.task_success, 4),
        "cost_saved_usd": round(full.cost_per_task_usd - ablated.cost_per_task_usd, 6),
        "latency_saved_ms": float(full.p95_latency_ms - ablated.p95_latency_ms),
        "attack_edges_removed": float(full.attack_edges - ablated.attack_edges),
        "operator_burden_removed": round(full.operator_burden - ablated.operator_burden, 2),
    }


def fixture_injections() -> list[InjectionResult]:
    """Return ten capstone failure-game outcomes."""

    rows = (
        ("crash_after_effect", True, True, True, "test_replay_records_one_effect"),
        ("stale_approval", True, True, True, "test_approval_hash_must_match"),
        ("duplicate_webhook", True, True, True, "test_duplicate_delivery_is_idempotent"),
        ("indirect_prompt_injection", True, True, True, "test_retrieved_text_cannot_select_tool"),
        ("memory_poison_write", True, True, True, "test_memory_write_requires_policy"),
        ("provider_outage_illegal_fallback", True, True, True, "test_fallback_preserves_region_policy"),
        ("judge_drift", True, True, True, "test_eval_anchor_set_blocks_release"),
        ("denial_of_wallet_loop", True, True, True, "test_loop_budget_terminates"),
        ("cross_tenant_retrieval", True, True, True, "test_acl_filter_precedes_similarity"),
        ("half_applied_migration", False, True, False, "test_migration_resume_reconciles_state"),
    )
    return [InjectionResult(*row) for row in rows]


def failure_summary(results: Iterable[InjectionResult]) -> dict[str, object]:
    """Aggregate detected, contained, and recovered failure outcomes."""

    materialized = list(results)
    if not materialized:
        raise ValueError("results cannot be empty")
    return {
        "injections": len(materialized),
        "detected": sum(result.detected for result in materialized),
        "contained": sum(result.contained for result in materialized),
        "recovered": sum(result.recovered for result in materialized),
        "needs_postmortem": [result.name for result in materialized if not result.recovered],
        "regression_tests": [result.regression_test for result in materialized],
    }


def build_report() -> dict[str, object]:
    """Return the ladder, decisions, and failure-game evidence packet."""

    ladder = fixture_ladder()
    return {
        "ladder": [
            {
                **asdict(report),
                "task_success": report.task_success,
                "pass_pow_4": round(report.pass_pow_k(4), 4),
                "operator_burden": report.operator_burden,
            }
            for report in ladder
        ],
        "decisions": earn_decisions(ladder),
        "failure_game": failure_summary(fixture_injections()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(build_report(), indent=2))
