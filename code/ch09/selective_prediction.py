"""Calibration, abstention, and uncertainty probes for Chapter 9."""

from __future__ import annotations

import math


def qa_fixture(offset: int = 0) -> list[dict[str, object]]:
    """Create calibrated-score fixtures with known semantic answer labels."""
    items = []
    for confidence, correct_count in zip((0.55, 0.65, 0.75, 0.85, 0.95), (10, 12, 14, 16, 17)):
        for j in range(20):
            correct = (j + offset) % 20 < correct_count
            rest = 1.0 - confidence
            probs = ([confidence, rest / 3, rest / 3, rest / 3] if correct else
                     [confidence, rest * 0.55, rest * 0.225, rest * 0.225])
            items.append({"confidence": confidence, "correct": correct, "probs": probs,
                          "true_index": 0 if correct else 1})
    return items


def fit_temperature(items: list[dict[str, object]]) -> float:
    """Fit one held-out confidence temperature by grid-searching binary NLL."""
    def nll(tau: float) -> float:
        total = 0.0
        for item in items:
            q = float(item["confidence"])
            calibrated = 1 / (1 + math.exp(-math.log(q / (1 - q)) / tau))
            p = calibrated if item["correct"] else 1 - calibrated
            total -= math.log(max(p, 1e-12))
        return total / len(items)
    return min((0.5 + i * 0.01 for i in range(451)), key=nll)


def calibrated_score(score: float, tau: float) -> float:
    """Apply scalar temperature calibration to a binary confidence."""
    return 1 / (1 + math.exp(-math.log(score / (1 - score)) / tau))


def calibration_metrics(items: list[dict[str, object]], tau: float = 1.0) -> dict[str, float]:
    """Compute binary Brier score, NLL, and ten-bin ECE."""
    rows, bins = [], [[] for _ in range(10)]
    for item in items:
        q = calibrated_score(float(item["confidence"]), tau)
        y = float(bool(item["correct"]))
        rows.append((q, y))
        bins[min(9, int(q * 10))].append((q, y))
    return {"brier": sum((q - y) ** 2 for q, y in rows) / len(rows),
            "nll": -sum(y * math.log(q) + (1 - y) * math.log(1 - q) for q, y in rows) / len(rows),
            "ece": sum(len(group) / len(rows) * abs(sum(q for q, _ in group) / len(group) -
                       sum(y for _, y in group) / len(group)) for group in bins if group)}


def reliability(items: list[dict[str, object]], tau: float) -> list[dict[str, float]]:
    """Aggregate accuracy and raw/calibrated confidence by fixture bin."""
    rows = []
    for raw in sorted({float(item["confidence"]) for item in items}):
        group = [item for item in items if item["confidence"] == raw]
        rows.append({"raw_confidence": raw, "calibrated_confidence": calibrated_score(raw, tau),
                     "accuracy": sum(bool(item["correct"]) for item in group) / len(group)})
    return rows


def risk_curve(items: list[dict[str, object]]) -> list[dict[str, float]]:
    """Report coverage, conditional selective risk, and marginal error."""
    rows = []
    for threshold in sorted({float(item["confidence"]) for item in items}, reverse=True):
        answered = [item for item in items if float(item["confidence"]) >= threshold]
        errors = sum(not bool(item["correct"]) for item in answered)
        rows.append({"threshold": threshold, "coverage": len(answered) / len(items),
                     "selective_risk": errors / len(answered), "marginal_error": errors / len(items)})
    return rows


def crc_threshold(items: list[dict[str, object]], alpha: float) -> float:
    """Choose maximum coverage under the binary conformal-risk correction."""
    for threshold in sorted({float(item["confidence"]) for item in items}):
        errors = sum(float(item["confidence"]) >= threshold and not bool(item["correct"]) for item in items)
        if (errors + 1) / (len(items) + 1) <= alpha:
            return threshold
    return math.inf


def split_conformal(calibration: list[dict[str, object]], test: list[dict[str, object]], alpha: float) -> dict[str, float]:
    """Build answer sets, then abstain unless the conformal set is a singleton."""
    scores = sorted(1 - list(item["probs"])[int(item["true_index"])] for item in calibration)
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha))) - 1
    qhat, contained, singleton, errors = scores[rank], 0, 0, 0
    for item in test:
        answer_set = [i for i, p in enumerate(item["probs"]) if 1 - p <= qhat + 1e-12]
        contained += int(int(item["true_index"]) in answer_set)
        if len(answer_set) == 1:
            singleton += 1
            errors += int(answer_set[0] != int(item["true_index"]))
    return {"qhat": qhat, "set_coverage": contained / len(test),
            "singleton_coverage": singleton / len(test),
            "singleton_risk": errors / singleton if singleton else 0.0}


def entropy(values: list[float]) -> float:
    """Return categorical entropy in nats."""
    return -sum(value * math.log(value) for value in values if value)


def semantic_entropy_probe() -> dict[str, float]:
    """Compare surface entropy with entropy after grouping paraphrases."""
    surface = [0.36, 0.24, 0.20, 0.12, 0.08]
    meanings = [surface[0] + surface[1], surface[2], surface[3], surface[4]]
    return {"surface_entropy_nats": entropy(surface), "semantic_entropy_nats": entropy(meanings)}
