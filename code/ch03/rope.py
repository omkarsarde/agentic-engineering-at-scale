"""Rotary position embeddings and inspectable context-extension scalings."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def ntk_base(base: float, factor: float, dim: int) -> float:
    """Return the base-frequency adjustment called NTK-aware scaling."""

    if dim <= 2 or factor < 1:
        raise ValueError("NTK scaling requires dim > 2 and factor >= 1")
    return base * factor ** (dim / (dim - 2))


def _correction_index(
    rotations: float, dim: int, base: float, original_context: int
) -> float:
    return dim * math.log(original_context / (rotations * 2 * math.pi)) / (
        2 * math.log(base)
    )


def inverse_frequencies(
    dim: int,
    *,
    method: str = "rope",
    base: float = 10_000.0,
    factor: float = 1.0,
    original_context: int = 2_048,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[Tensor, float]:
    """Return frequency bands and the YaRN query/key magnitude multiplier."""

    if dim % 2 or dim < 4 or factor < 1:
        raise ValueError("dim must be even and >= 4; factor must be >= 1")
    method = method.casefold()
    indices = torch.arange(0, dim, 2, dtype=torch.float64)
    ordinary = base ** (-indices / dim)
    if method == "rope":
        return ordinary, 1.0
    if method == "pi":
        return ordinary / factor, 1.0
    if method == "ntk":
        adjusted = ntk_base(base, factor, dim)
        return adjusted ** (-indices / dim), 1.0
    if method != "yarn":
        raise ValueError(f"unknown RoPE method: {method}")

    interpolated = ordinary / factor
    low = math.floor(_correction_index(beta_fast, dim, base, original_context))
    high = math.ceil(_correction_index(beta_slow, dim, base, original_context))
    low = max(low, 0)
    high = min(high, dim // 2 - 1)
    if low == high:
        high += 1e-3
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    extrapolation_weight = 1 - ramp
    frequencies = interpolated * (1 - extrapolation_weight) + ordinary * extrapolation_weight
    magnitude = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return frequencies, magnitude


def apply_rope(
    vectors: Tensor,
    positions: Tensor,
    frequencies: Tensor,
    magnitude: float = 1.0,
) -> Tensor:
    """Rotate interleaved coordinate pairs in [..., sequence, dimension]."""

    if vectors.size(-1) != 2 * frequencies.numel():
        raise ValueError("frequency count must be half the vector dimension")
    if vectors.size(-2) != positions.numel():
        raise ValueError("one position is required per sequence element")
    angles = positions.to(frequencies.dtype).unsqueeze(-1) * frequencies
    shape = (1,) * (vectors.ndim - 2) + angles.shape
    cosine = angles.cos().reshape(shape).to(vectors)
    sine = angles.sin().reshape(shape).to(vectors)
    even, odd = vectors[..., 0::2], vectors[..., 1::2]
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2) * magnitude
