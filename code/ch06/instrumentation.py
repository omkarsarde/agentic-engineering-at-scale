"""Byte and throughput instrumentation shared by the Chapter 6 probe."""

from __future__ import annotations

import time

import torch
from torch.distributed.tensor import DTensor


def local_bytes(tensor: torch.Tensor | None) -> int:
    """Return storage bytes owned by this rank, including DTensor shards."""

    if tensor is None:
        return 0
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.numel() * local.element_size()


def local_squared_sum(tensor: torch.Tensor | None) -> float:
    """Return the rank-local squared sum for a Tensor or DTensor shard."""

    if tensor is None:
        return 0.0
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return float(local.detach().double().square().sum())


def matmul_roofline(device: torch.device) -> float:
    """Measure a local matmul reference rate for an environment-specific MFU proxy."""

    size, repeats = 384, 4
    left = torch.randn(size, size, device=device)
    right = torch.randn(size, size, device=device)
    for _ in range(2):
        left @ right
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        left @ right
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 2 * size**3 * repeats / (time.perf_counter() - start)
