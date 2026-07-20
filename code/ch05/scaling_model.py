"""Fit the joint parameter--data scaling surface used in Chapter 5."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PARAMETER_REFERENCE = 100_000_000.0
TOKEN_REFERENCE = 2_000_000_000.0


@dataclass(frozen=True)
class ScalingObservation:
    """One base-model run used to fit a scaling surface."""

    parameters: float
    tokens: float
    loss: float

    @property
    def compute_flops(self) -> float:
        """Return the dense-Transformer approximation ``6ND``."""

        return 6.0 * self.parameters * self.tokens


@dataclass(frozen=True)
class ScalingLaw:
    """A fitted law ``E + A(N/N0)^-alpha + B(D/D0)^-beta``."""

    irreducible_loss: float
    parameter_coefficient: float
    data_coefficient: float
    parameter_exponent: float
    data_exponent: float
    parameter_reference: float = PARAMETER_REFERENCE
    token_reference: float = TOKEN_REFERENCE

    def loss(self, parameters: float | np.ndarray, tokens: float | np.ndarray) -> np.ndarray:
        """Predict loss for positive absolute parameter and token counts."""

        n = np.asarray(parameters, dtype=float) / self.parameter_reference
        d = np.asarray(tokens, dtype=float) / self.token_reference
        if np.any(n <= 0) or np.any(d <= 0):
            raise ValueError("parameters and tokens must be positive")
        return (
            self.irreducible_loss
            + self.parameter_coefficient * n ** (-self.parameter_exponent)
            + self.data_coefficient * d ** (-self.data_exponent)
        )

    def compute_optimal(self, compute_flops: float) -> tuple[float, float, float]:
        """Return compute-optimal ``(N, D, loss)`` under ``C = 6ND``."""

        if compute_flops <= 0:
            raise ValueError("compute_flops must be positive")
        scaled_compute = compute_flops / (
            6.0 * self.parameter_reference * self.token_reference
        )
        ratio = (
            self.parameter_exponent
            * self.parameter_coefficient
            / (self.data_exponent * self.data_coefficient)
        )
        n = (ratio * scaled_compute**self.data_exponent) ** (
            1.0 / (self.parameter_exponent + self.data_exponent)
        )
        d = scaled_compute / n
        parameters = n * self.parameter_reference
        tokens = d * self.token_reference
        return parameters, tokens, float(self.loss(parameters, tokens))

    def as_dict(self) -> dict[str, float]:
        """Return coefficients in a JSON-serializable mapping."""

        return {key: float(value) for key, value in asdict(self).items()}


def load_observations(path: Path) -> list[ScalingObservation]:
    """Load ``parameters,tokens,loss`` observations from CSV."""

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ScalingObservation(float(row["parameters"]), float(row["tokens"]), float(row["loss"]))
            for row in csv.DictReader(handle)
        ]


def _levenberg_marquardt(
    residual: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    *,
    iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize a small residual vector without an external optimizer."""

    theta = start.astype(float).copy()
    errors = residual(theta)
    damping = 1e-3
    for _ in range(iterations):
        jacobian = np.empty((errors.size, theta.size))
        for column in range(theta.size):
            step = 1e-5 * max(1.0, abs(theta[column]))
            offset = np.zeros_like(theta)
            offset[column] = step
            jacobian[:, column] = (
                residual(theta + offset) - residual(theta - offset)
            ) / (2 * step)
        normal = jacobian.T @ jacobian + damping * np.eye(theta.size)
        gradient = jacobian.T @ errors
        try:
            delta = np.linalg.solve(normal, -gradient)
        except np.linalg.LinAlgError:
            damping *= 10
            continue
        candidate = np.clip(theta + delta, -12.0, 5.0)
        candidate_errors = residual(candidate)
        if np.sum(candidate_errors**2) < np.sum(errors**2):
            theta, errors = candidate, candidate_errors
            damping = max(1e-10, damping / 2)
            if np.linalg.norm(delta) < 1e-9:
                break
        else:
            damping = min(1e12, damping * 4)
    return theta, errors


def fit_scaling_law(observations: Iterable[ScalingObservation]) -> ScalingLaw:
    """Fit a positive five-parameter surface by robust least squares."""

    rows = list(observations)
    if len(rows) < 6:
        raise ValueError("at least six observations are required")
    if any(
        not np.isfinite(value) or value <= 0
        for row in rows
        for value in (row.parameters, row.tokens, row.loss)
    ):
        raise ValueError("parameters, tokens, and losses must be finite and positive")
    n = np.asarray([row.parameters / PARAMETER_REFERENCE for row in rows])
    d = np.asarray([row.tokens / TOKEN_REFERENCE for row in rows])
    y = np.asarray([row.loss for row in rows])
    if np.unique(n).size < 2 or np.unique(d).size < 2:
        raise ValueError("the ladder must vary both parameters and tokens")
    ceiling = float(y.min() - 1e-5)

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float, float]:
        floor = ceiling - np.exp(theta[0])
        return floor, *(float(np.exp(value)) for value in theta[1:])

    def residual(theta: np.ndarray) -> np.ndarray:
        floor, a, b, alpha, beta = unpack(theta)
        return floor + a * n ** (-alpha) + b * d ** (-beta) - y

    starts = (
        (1.4, 0.8, 1.0, 0.34, 0.28),
        (0.8, 1.5, 1.5, 0.25, 0.25),
        (1.8, 0.4, 0.8, 0.5, 0.35),
    )
    best_theta: np.ndarray | None = None
    best_residual: np.ndarray | None = None
    for floor, a, b, alpha, beta in starts:
        theta = np.log([max(ceiling - floor, 1e-4), a, b, alpha, beta])
        theta, errors = _levenberg_marquardt(residual, theta)
        if best_residual is None or np.sum(errors**2) < np.sum(best_residual**2):
            best_theta, best_residual = theta, errors
    assert best_theta is not None
    return ScalingLaw(*unpack(best_theta))
