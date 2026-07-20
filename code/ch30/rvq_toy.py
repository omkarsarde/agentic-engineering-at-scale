"""Tiny residual-vector-quantization demonstration for Chapter 30."""

from __future__ import annotations

from math import sqrt


Codebook = tuple[tuple[float, ...], ...]


def squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return squared Euclidean distance."""

    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def residual_quantize(vector: tuple[float, ...], codebooks: tuple[Codebook, ...]) -> list[float]:
    """Return reconstruction errors after each residual codebook."""

    reconstruction = tuple(0.0 for _ in vector)
    errors: list[float] = []
    for codebook in codebooks:
        residual = tuple(value - estimate for value, estimate in zip(vector, reconstruction, strict=True))
        chosen = min(codebook, key=lambda code: squared_distance(residual, code))
        reconstruction = tuple(value + delta for value, delta in zip(reconstruction, chosen, strict=True))
        errors.append(sqrt(squared_distance(vector, reconstruction)))
    return errors


def demo() -> list[float]:
    """Quantize one latent with progressively finer codebooks."""

    vector = (0.82, -0.36)
    codebooks: tuple[Codebook, ...] = (
        ((0.5, -0.5), (0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5)),
        ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)),
        ((0.0625, 0.125), (-0.0625, -0.125), (0.125, 0.0), (-0.125, 0.0)),
    )
    return residual_quantize(vector, codebooks)


if __name__ == "__main__":
    print(demo())
