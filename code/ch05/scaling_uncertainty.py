"""Residual-bootstrap intervals for an extrapolated scaling-law optimum."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from scaling_model import ScalingObservation, fit_scaling_law


def extrapolation_interval(
    observations: Iterable[ScalingObservation],
    *,
    compute_multiplier: float = 10.0,
    bootstrap_samples: int = 200,
    seed: int = 5,
) -> dict[str, float]:
    """Return residual-bootstrap fit and predictive intervals at larger compute."""

    rows = list(observations)
    if compute_multiplier <= 0 or bootstrap_samples < 20:
        raise ValueError("use a positive multiplier and at least 20 bootstrap samples")
    target_compute = max(row.compute_flops for row in rows) * compute_multiplier
    fitted = fit_scaling_law(rows)
    target_n, target_d, target_loss = fitted.compute_optimal(target_compute)
    fitted_losses = np.asarray(
        [float(fitted.loss(row.parameters, row.tokens)) for row in rows]
    )
    residuals = np.asarray([row.loss for row in rows]) - fitted_losses
    residuals -= residuals.mean()
    generator = np.random.default_rng(seed)
    fitted_predictions: list[float] = []
    predictive_draws: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_residuals = generator.choice(residuals, size=len(rows), replace=True)
        sample = [
            ScalingObservation(row.parameters, row.tokens, prediction + noise)
            for row, prediction, noise in zip(rows, fitted_losses, sampled_residuals)
        ]
        try:
            law = fit_scaling_law(sample)
            _, _, prediction = law.compute_optimal(target_compute)
        except (ValueError, FloatingPointError, OverflowError):
            continue
        predictive = prediction + float(generator.choice(residuals))
        if np.isfinite(prediction) and np.isfinite(predictive):
            fitted_predictions.append(prediction)
            predictive_draws.append(predictive)
    if len(predictive_draws) < bootstrap_samples // 2:
        raise RuntimeError("too few valid bootstrap fits; expand the ladder")
    fit_low, fit_high = np.quantile(fitted_predictions, (0.05, 0.95))
    prediction_low, prediction_high = np.quantile(predictive_draws, (0.05, 0.95))
    return {
        "target_compute_flops": target_compute,
        "optimal_parameters": target_n,
        "optimal_tokens": target_d,
        "predicted_loss": target_loss,
        "fit_p05": float(fit_low),
        "fit_p95": float(fit_high),
        "prediction_p05": float(prediction_low),
        "prediction_p95": float(prediction_high),
        "valid_bootstrap_fits": float(len(predictive_draws)),
    }
