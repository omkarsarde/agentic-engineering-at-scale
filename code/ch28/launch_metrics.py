"""Build a deterministic quantitative appendix for an agent launch review."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FIXTURE = Path(__file__).with_name("fixtures") / "expense_triage.json"
QUADRANTS = ("correct_accept", "correct_reject", "wrong_accept", "wrong_reject")
STAGES = (
    "exposed",
    "invoked",
    "completed",
    "accepted_or_edited",
    "outcome_realized",
    "retained",
)


def load_fixture(path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    """Load the launch-review fixture.

    Args:
        path: JSON file containing cohort counts, costs, gates, and risk metadata.

    Returns:
        Parsed fixture dictionary.

    Raises:
        ValueError: If cohort counts violate the declared log shape.
    """
    fixture = json.loads(path.read_text(encoding="utf-8"))
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture: dict[str, Any]) -> None:
    """Reject inconsistent cohort, quadrant, or funnel counts."""
    for name, cohort in fixture["cohorts"].items():
        size = cohort["size"]
        if sum(cohort["quadrants"].values()) != size:
            raise ValueError(f"{name}: reliance quadrants do not sum to cohort size")
        counts = [cohort["funnel"][stage] for stage in STAGES]
        if counts[0] != size or any(left < right for left, right in zip(counts, counts[1:])):
            raise ValueError(f"{name}: funnel must begin at size and never increase")
        if cohort["confirmed_catches"] > cohort["quadrants"]["wrong_reject"]:
            raise ValueError(f"{name}: confirmed catches exceed wrong rejections")


def expand_logs(fixture: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand compact cohort counts into two deterministic 500-row logs."""
    decisions: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    case_id = 0
    for cohort_name, cohort in fixture["cohorts"].items():
        for quadrant in QUADRANTS:
            for offset in range(cohort["quadrants"][quadrant]):
                decisions.append(
                    {
                        "case_id": case_id,
                        "cohort": cohort_name,
                        "quadrant": quadrant,
                        "confirmed_catch": quadrant == "wrong_reject"
                        and offset < cohort["confirmed_catches"],
                        "review_seconds": cohort["review_seconds"],
                    }
                )
                case_id += 1
        for index in range(cohort["size"]):
            usage.append(
                {
                    "cohort": cohort_name,
                    **{stage: index < cohort["funnel"][stage] for stage in STAGES},
                }
            )
    return decisions, usage


def _reliance_rates(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compute reliance rates for one slice."""
    rows = list(rows)
    counts = Counter(row["quadrant"] for row in rows)
    correct = counts["correct_accept"] + counts["correct_reject"]
    wrong = counts["wrong_accept"] + counts["wrong_reject"]
    catches = sum(row["confirmed_catch"] for row in rows)
    return {
        "n": len(rows),
        "quadrants": {name: counts[name] for name in QUADRANTS},
        "appropriate_rate": round((counts["correct_accept"] + counts["wrong_reject"]) / len(rows), 4),
        "accept_when_correct": round(counts["correct_accept"] / correct, 4),
        "reject_when_wrong": round(counts["wrong_reject"] / wrong, 4),
        "overreliance_rate": round(counts["wrong_accept"] / wrong, 4),
        "underreliance_rate": round(counts["correct_reject"] / correct, 4),
        "confirmed_catch_rate": round(catches / counts["wrong_reject"], 4),
    }


def reliance_report(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall and cohort reliance tables."""
    cohorts = sorted({row["cohort"] for row in decisions})
    return {
        "overall": _reliance_rates(decisions),
        "by_cohort": {
            name: _reliance_rates(row for row in decisions if row["cohort"] == name)
            for name in cohorts
        },
    }


def _funnel_slice(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count nested funnel stages and their adjacent conversions."""
    rows = list(rows)
    counts = {stage: sum(row[stage] for row in rows) for stage in STAGES}
    conversions = {STAGES[0]: 1.0}
    for prior, stage in zip(STAGES, STAGES[1:]):
        conversions[stage] = round(counts[stage] / counts[prior], 4)
    return {"counts": counts, "conversion_from_previous": conversions}


def funnel_report(usage: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the invocation-to-outcome funnel overall and by cohort."""
    cohorts = sorted({row["cohort"] for row in usage})
    return {
        "overall": _funnel_slice(usage),
        "by_cohort": {
            name: _funnel_slice(row for row in usage if row["cohort"] == name)
            for name in cohorts
        },
    }


def economics_report(
    fixture: dict[str, Any], decisions: list[dict[str, Any]], usage: list[dict[str, Any]]
) -> dict[str, Any]:
    """Price accepted outcomes, review labor, remediation, and allocated energy."""
    costs = fixture["economics"]
    outcomes = sum(row["outcome_realized"] for row in usage)
    review_seconds = sum(row["review_seconds"] for row in decisions)
    wrong_accepts = sum(row["quadrant"] == "wrong_accept" for row in decisions)
    machine = len(usage) * (costs["inference_per_exposure"] + costs["retrieval_per_exposure"])
    review = review_seconds * costs["review_hourly_rate"] / 3600.0
    remediation = wrong_accepts * costs["remediation_per_wrong_accept"]
    non_review = machine + remediation
    total = non_review + review
    max_average_review = (
        (costs["baseline_cost_per_outcome"] * outcomes - non_review)
        * 3600.0
        / (costs["review_hourly_rate"] * len(decisions))
    )
    energy_wh = len(usage) * costs["allocated_energy_wh_per_exposure"]
    return {
        "outcomes": outcomes,
        "cost_components": {
            "inference_and_retrieval": round(machine, 2),
            "review_labor": round(review, 2),
            "remediation": round(remediation, 2),
        },
        "total_cost": round(total, 2),
        "cost_per_outcome": round(total / outcomes, 4),
        "baseline_cost_per_outcome": costs["baseline_cost_per_outcome"],
        "average_review_seconds": round(review_seconds / len(decisions), 1),
        "break_even_review_seconds": round(max_average_review, 1),
        "allocated_energy_wh_per_outcome": round(energy_wh / outcomes, 4),
        "energy_boundary": costs["energy_boundary"],
    }


def evaluate_gates(
    fixture: dict[str, Any], reliance: dict[str, Any], funnel: dict[str, Any], economics: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate predeclared product, reliance, accessibility, and economic gates."""
    limits = fixture["kill_criteria"]
    max_cohort = max(item["overreliance_rate"] for item in reliance["by_cohort"].values())
    assistive_completion = funnel["by_cohort"]["assistive_technology"]["conversion_from_previous"]["completed"]
    checks = (
        ("appropriate_reliance", reliance["overall"]["appropriate_rate"], limits["minimum_appropriate_rate"], ">="),
        ("cohort_overreliance", max_cohort, limits["maximum_cohort_overreliance"], "<="),
        ("assistive_completion", assistive_completion, limits["minimum_assistive_completion"], ">="),
        ("cost_per_outcome", economics["cost_per_outcome"], economics["baseline_cost_per_outcome"], "<="),
    )
    return [
        {
            "gate": name,
            "observed": observed,
            "threshold": threshold,
            "operator": operator,
            "passed": observed >= threshold if operator == ">=" else observed <= threshold,
        }
        for name, observed, threshold, operator in checks
    ]


def build_launch_review(fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the launch packet's deterministic quantitative appendix."""
    fixture = fixture or load_fixture()
    validate_fixture(fixture)
    decisions, usage = expand_logs(fixture)
    reliance = reliance_report(decisions)
    funnel = funnel_report(usage)
    economics = economics_report(fixture, decisions, usage)
    gates = evaluate_gates(fixture, reliance, funnel, economics)
    failed = [gate["gate"] for gate in gates if not gate["passed"]]
    return {
        "system": fixture["system"],
        "log_rows": {"reviewer_decisions": len(decisions), "usage_ledger": len(usage)},
        "risk_register": fixture["risk_register"],
        "reliance": reliance,
        "funnel": funnel,
        "economics": economics,
        "gates": gates,
        "recommendation": "SHIP" if not failed else "HOLD_FULL_LAUNCH",
        "failed_conditions": failed,
        "permitted_next_step": fixture["permitted_next_step"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--enforce", action="store_true", help="exit 2 when launch gates fail")
    args = parser.parse_args()
    report = build_launch_review(load_fixture(args.fixture))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if args.enforce and report["recommendation"] != "SHIP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
