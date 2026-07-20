"""Translate a training-compute budget into time, cost, power, and token economics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostAssumptions:
    """Explicit, replaceable assumptions for one hypothetical training run."""

    parameters: float = 30e9
    tokens: float = 600e9
    accelerators: int = 8_192
    peak_tflops_each: float = 1_000.0
    mfu: float = 0.40
    hourly_price_each: float = 4.0
    facility_kw_each: float = 1.2


def evaluate_cost(assumptions: CostAssumptions) -> dict[str, float]:
    """Evaluate dense ``C=6ND`` economics with MFU and facility power."""

    values = asdict(assumptions)
    if any(value <= 0 for value in values.values()) or assumptions.mfu > 1:
        raise ValueError("cost assumptions must be positive and MFU must not exceed one")
    compute = 6 * assumptions.parameters * assumptions.tokens
    useful_flops_per_second_each = assumptions.peak_tflops_each * 1e12 * assumptions.mfu
    accelerator_hours = compute / useful_flops_per_second_each / 3_600
    wall_hours = accelerator_hours / assumptions.accelerators
    cost = accelerator_hours * assumptions.hourly_price_each
    energy_mwh = accelerator_hours * assumptions.facility_kw_each / 1_000
    return {
        **values,
        "training_flops": compute,
        "accelerator_hours": accelerator_hours,
        "wall_days": wall_hours / 24,
        "training_cost": cost,
        "cost_per_million_tokens": cost / assumptions.tokens * 1e6,
        "average_facility_mw": assumptions.accelerators * assumptions.facility_kw_each / 1_000,
        "energy_mwh": energy_mwh,
    }


def mfu_sensitivity(
    base: CostAssumptions, mfus: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
) -> list[dict[str, float]]:
    """Evaluate one run over an MFU grid while holding every other input fixed."""

    rows = []
    for mfu in mfus:
        candidate = CostAssumptions(**{**asdict(base), "mfu": mfu})
        rows.append(evaluate_cost(candidate))
    return rows
