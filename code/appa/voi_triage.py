"""One-step value-of-information calculation for Appendix A."""

from __future__ import annotations


def best_expected_loss(probability_urgent: float, false_negative: float, false_positive: float) -> float:
    """Return minimum expected loss across escalate and do-not-escalate actions."""

    escalate = (1 - probability_urgent) * false_positive
    do_not_escalate = probability_urgent * false_negative
    return min(escalate, do_not_escalate)


def value_of_signal(
    prior_urgent: float,
    sensitivity: float,
    specificity: float,
    false_negative: float,
    false_positive: float,
) -> float:
    """Return expected loss avoided by observing one binary signal."""

    prior_loss = best_expected_loss(prior_urgent, false_negative, false_positive)
    positive = prior_urgent * sensitivity + (1 - prior_urgent) * (1 - specificity)
    posterior_positive = prior_urgent * sensitivity / positive
    posterior_negative = prior_urgent * (1 - sensitivity) / (1 - positive)
    observed_loss = positive * best_expected_loss(
        posterior_positive, false_negative, false_positive
    ) + (1 - positive) * best_expected_loss(
        posterior_negative, false_negative, false_positive
    )
    return prior_loss - observed_loss


if __name__ == "__main__":
    print(value_of_signal(0.2, 0.85, 0.9, 10.0, 1.0))
