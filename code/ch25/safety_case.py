"""Deterministic interpretability diagnostics and a fail-closed AI safety case."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CASE = Path(__file__).with_name("fixtures") / "deployment.json"
DIAGNOSTIC_KINDS = {"logit_lens", "linear_probe", "activation_steering", "sae_feature"}


def residual_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic residual states, binary labels, and a known risk direction."""
    rows, layers, width = 48, 6, 8
    row = np.arange(rows, dtype=float)[:, None]
    col = np.arange(width, dtype=float)[None, :]
    labels = np.where(np.arange(rows) % 2 == 0, 1.0, -1.0)
    direction = np.array([1.0, -0.7, 0.4, 0.0, 0.2, -0.3, 0.1, 0.5])
    direction /= np.linalg.norm(direction)
    base = 0.42 * np.sin((row + 1.0) * (col + 1.0) * 0.37)
    base += 0.31 * np.cos((row + 2.0) * (col + 1.0) * 0.19)
    strengths = np.array([0.00, 0.08, 0.22, 0.48, 0.82, 1.18])
    residuals = np.stack(
        [base + strength * labels[:, None] * direction for strength in strengths],
        axis=1,
    )
    return residuals, labels, direction


def _ridge_probe(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """Fit a small ridge probe with a closed-form deterministic solve."""
    design = np.column_stack([train_x, np.ones(len(train_x))])
    penalty = np.eye(design.shape[1]) * 0.25
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ train_y)


def _accuracy(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    design = np.column_stack([features, np.ones(len(features))])
    predictions = np.where(design @ weights >= 0.0, 1.0, -1.0)
    return float(np.mean(predictions == labels))


def interpretability_diagnostics() -> dict[str, Any]:
    """Measure a lens, probe selectivity, and a steering dose response."""
    residuals, labels, direction = residual_fixture()
    lens_margin = [
        float(np.mean(labels * (residuals[:, layer, :] @ direction)))
        for layer in range(residuals.shape[1])
    ]

    train = np.arange(32)
    test = np.arange(32, 48)
    layer = 4
    probe = _ridge_probe(residuals[train, layer, :], labels[train])
    probe_accuracy = _accuracy(probe, residuals[test, layer, :], labels[test])

    rng = np.random.default_rng(25)
    control_labels = labels[rng.permutation(len(labels))]
    control_probe = _ridge_probe(residuals[train, layer, :], control_labels[train])
    control_accuracy = _accuracy(
        control_probe, residuals[test, layer, :], control_labels[test]
    )

    risky = residuals[labels == 1.0, -1, :]
    steering = []
    for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
        steered = risky - alpha * direction
        risk_rate = float(np.mean((steered @ direction) > 0.0))
        cosine = np.sum(risky * steered, axis=1) / (
            np.linalg.norm(risky, axis=1) * np.linalg.norm(steered, axis=1)
        )
        steering.append(
            {
                "alpha": alpha,
                "risk_rate": round(risk_rate, 4),
                "utility_proxy": round(float(np.mean(cosine)), 4),
            }
        )

    return {
        "fixture": "synthetic residual-stream states; not a deployed model",
        "lens_margin": [round(value, 4) for value in lens_margin],
        "probe": {
            "layer": layer,
            "accuracy": round(probe_accuracy, 4),
            "control_accuracy": round(control_accuracy, 4),
            "selectivity": round(probe_accuracy - control_accuracy, 4),
        },
        "steering": steering,
    }


def load_case(path: Path = DEFAULT_CASE) -> dict[str, Any]:
    """Load a deployment safety-case fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic_evidence(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert internal measurements into scoped, non-dispositive evidence items."""
    probe = diagnostics["probe"]
    steering = diagnostics["steering"]
    return [
        {
            "id": "E-LENS",
            "kind": "logit_lens",
            "status": "pass",
            "scope": "Synthetic layerwise risk-token margin",
            "value": diagnostics["lens_margin"][-1],
            "limitation": "A decoded margin does not identify a causal computation.",
        },
        {
            "id": "E-PROBE",
            "kind": "linear_probe",
            "status": "pass" if probe["selectivity"] >= 0.25 else "fail",
            "scope": "Held-out risk-label selectivity at one residual layer",
            "value": probe["selectivity"],
            "limitation": "Decodability does not prove that the policy uses the feature.",
        },
        {
            "id": "E-STEER",
            "kind": "activation_steering",
            "status": "pass" if steering[-1]["risk_rate"] < steering[0]["risk_rate"] else "fail",
            "scope": "Synthetic intervention along the inspected direction",
            "value": steering[-1]["risk_rate"],
            "limitation": "A local intervention may create off-target effects or fail out of distribution.",
        },
    ]


def validate_case(case: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    """Reject malformed claims and dangling evidence references."""
    claim_ids = [claim["id"] for claim in case["claims"]]
    evidence_ids = [item["id"] for item in evidence]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("claim IDs must be unique")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be unique")
    known = set(evidence_ids)
    for claim in case["claims"]:
        if claim["argument"] not in {"inability", "control", "trustworthiness"}:
            raise ValueError(f"unknown argument type: {claim['argument']}")
        referenced = list(claim.get("evidence", []))
        for control in claim.get("controls", []):
            referenced.extend(control.get("evidence", []))
        missing = sorted(set(referenced) - known)
        if missing:
            raise ValueError(f"{claim['id']} references missing evidence: {missing}")


def _passing(evidence_by_id: dict[str, dict[str, Any]], ids: list[str]) -> bool:
    return bool(ids) and all(evidence_by_id[item]["status"] == "pass" for item in ids)


def evaluate_claim(
    claim: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate one scoped safety argument against explicit evidence rules."""
    reasons: list[str] = []
    argument = claim["argument"]

    if argument == "control":
        controls = claim.get("controls", [])
        if not controls:
            reasons.append("no control is named")
        for control in controls:
            ids = control.get("evidence", [])
            kinds = {evidence_by_id[item]["kind"] for item in ids}
            if not _passing(evidence_by_id, ids):
                reasons.append(f"{control['id']} lacks passing evidence")
            if kinds and kinds <= DIAGNOSTIC_KINDS:
                reasons.append(f"{control['id']} is supported only by internal diagnostics")

    elif argument == "inability":
        capability = claim["capability"]
        if capability["score"] > capability["threshold"]:
            reasons.append("capability score exceeds the deployment threshold")
        if capability["elicitation_coverage"] < capability["required_coverage"]:
            reasons.append("elicitation coverage is below the declared minimum")
        if capability.get("sandbagging_check") != "pass":
            reasons.append("evaluation-integrity or sandbagging checks remain open")
        if not _passing(evidence_by_id, claim.get("evidence", [])):
            reasons.append("capability evidence is missing or failing")

    else:
        ids = claim.get("evidence", [])
        kinds = {evidence_by_id[item]["kind"] for item in ids}
        if not _passing(evidence_by_id, ids):
            reasons.append("trustworthiness evidence is missing or failing")
        if len(kinds - DIAGNOSTIC_KINDS) < 2:
            reasons.append("fewer than two independent non-diagnostic evidence classes")
        if claim.get("unresolved_counterevidence", True):
            reasons.append("counterevidence remains unresolved")

    return {
        "id": claim["id"],
        "severity": claim["severity"],
        "argument": argument,
        "status": "supported" if not reasons else "gap",
        "reasons": reasons,
    }


def build_safety_case(case: dict[str, Any] | None = None) -> dict[str, Any]:
    """Join internal diagnostics to a claim-evidence graph and decide deployment."""
    source = deepcopy(case if case is not None else load_case())
    diagnostics = interpretability_diagnostics()
    evidence = source["evidence"] + diagnostic_evidence(diagnostics)
    validate_case(source, evidence)
    by_id = {item["id"]: item for item in evidence}
    results = [evaluate_claim(claim, by_id) for claim in source["claims"]]
    blocking = [
        item for item in results
        if item["status"] == "gap" and item["severity"] in {"high", "critical"}
    ]
    return {
        "system": source["system"],
        "top_claim": source["top_claim"],
        "interpretability": diagnostics,
        "claims": results,
        "summary": {
            "supported": sum(item["status"] == "supported" for item in results),
            "gaps": sum(item["status"] == "gap" for item in results),
            "blocking_claims": [item["id"] for item in blocking],
        },
        "decision": "BLOCK" if blocking else "APPROVE",
    }


def plot_diagnostics(report: dict[str, Any]):
    """Plot the three synthetic measurements used by the chapter."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-25-inline"
    diagnostics = report["interpretability"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    axes[0].plot(range(len(diagnostics["lens_margin"])), diagnostics["lens_margin"], marker="o")
    axes[0].set(xlabel="residual layer", ylabel="mean risk-token margin (a.u.)")
    probe = diagnostics["probe"]
    axes[1].bar(["task", "shuffled control"], [probe["accuracy"], probe["control_accuracy"]])
    axes[1].set(ylabel="held-out accuracy", ylim=(0.0, 1.05))
    doses = diagnostics["steering"]
    axes[2].plot([item["alpha"] for item in doses], [item["risk_rate"] for item in doses], marker="o", label="risk rate")
    axes[2].plot([item["alpha"] for item in doses], [item["utility_proxy"] for item in doses], marker="s", linestyle="--", label="utility proxy")
    axes[2].set(xlabel="steering dose (alpha)", ylabel="fraction or cosine", ylim=(-0.05, 1.05))
    axes[2].legend(frameon=False)
    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce", action="store_true", help="exit 2 when the case blocks deployment")
    args = parser.parse_args()
    report = build_safety_case(load_case(args.case))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 2 if args.enforce and report["decision"] == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
