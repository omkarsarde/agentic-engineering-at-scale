"""Unit-explicit sizing helpers for Appendix B worksheets."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Mapping, Sequence


GIB = 1024**3


def training_memory_bytes(
    total_params: int,
    trainable_params: int,
    weight_bytes: float = 2,
    gradient_bytes: float = 2,
    master_weight_bytes: float = 4,
    optimizer_bytes: float = 8,
) -> float:
    """Estimate parameter-state memory; frozen parameters pay only weight bytes."""

    if not 0 <= trainable_params <= total_params:
        raise ValueError("trainable_params must lie within total_params")
    trainable_state = gradient_bytes + master_weight_bytes + optimizer_bytes
    return total_params * weight_bytes + trainable_params * trainable_state


def training_flops(params: int, tokens: int, factor: float = 6.0) -> float:
    """Return the dense-transformer first-order training-compute estimate."""

    if min(params, tokens, factor) <= 0:
        raise ValueError("params, tokens, and factor must be positive")
    return factor * params * tokens


def kv_cache_bytes(
    layers: int,
    kv_heads: int,
    head_dim: int,
    sequence_tokens: int,
    batch: int = 1,
    dtype_bytes: float = 2,
) -> float:
    """Return K-plus-V cache bytes for one decoder deployment."""

    values = (layers, kv_heads, head_dim, sequence_tokens, batch, dtype_bytes)
    if min(values) <= 0:
        raise ValueError("all KV dimensions must be positive")
    return 2 * layers * kv_heads * head_dim * sequence_tokens * batch * dtype_bytes


@dataclass(frozen=True)
class CapacityEstimate:
    """Little's-law work in progress with explicit headroom."""

    mean_inflight: float
    provisioned_inflight: int
    service_rate_per_worker: float
    workers: int


def capacity(
    arrival_per_second: float,
    fanout: float,
    residence_seconds: float,
    service_seconds_per_item: float,
    headroom: float = 1.3,
) -> CapacityEstimate:
    """Estimate in-flight work and worker count for a stable target load."""

    if min(arrival_per_second, fanout, residence_seconds, service_seconds_per_item) <= 0:
        raise ValueError("capacity inputs must be positive")
    if headroom < 1:
        raise ValueError("headroom must be at least one")
    mean = arrival_per_second * fanout * residence_seconds
    provisioned = int(mean * headroom + 0.999999)
    service_rate = 1 / service_seconds_per_item
    workers = int(arrival_per_second * fanout / service_rate * headroom + 0.999999)
    return CapacityEstimate(mean, provisioned, service_rate, workers)


def rag_storage_bytes(
    vectors: int,
    dimensions: int,
    dtype_bytes: float = 4,
    index_overhead: float = 1.5,
    replicas: int = 2,
) -> float:
    """Estimate vector payload plus index overhead across replicas."""

    if min(vectors, dimensions, dtype_bytes, index_overhead, replicas) <= 0:
        raise ValueError("RAG storage inputs must be positive")
    return vectors * dimensions * dtype_bytes * index_overhead * replicas


def media_gpu_seconds(
    megapixels: float,
    video_seconds: float = 0,
    candidates: int = 1,
    image_gpu_seconds_per_mp: float = 12,
    video_gpu_seconds_per_output_second: float = 4,
) -> float:
    """Return an illustrative accelerator-work estimate for one media request."""

    if megapixels < 0 or video_seconds < 0 or candidates <= 0:
        raise ValueError("media dimensions cannot be negative")
    per_candidate = megapixels * image_gpu_seconds_per_mp
    per_candidate += video_seconds * video_gpu_seconds_per_output_second
    return candidates * per_candidate


def pass_pow_k(success_probability: float, trajectory_steps: int) -> float:
    """Return the independence-model probability all steps succeed."""

    if not 0 <= success_probability <= 1 or trajectory_steps <= 0:
        raise ValueError("probability must be in [0,1] and steps positive")
    return success_probability**trajectory_steps


def critical_path_ms(
    durations_ms: Mapping[str, float], dependencies: Mapping[str, Sequence[str]]
) -> float:
    """Return longest dependency path in an acyclic task graph."""

    cache: dict[str, float] = {}
    visiting: set[str] = set()

    def finish(node: str) -> float:
        if node in cache:
            return cache[node]
        if node in visiting:
            raise ValueError("dependency graph contains a cycle")
        if node not in durations_ms:
            raise ValueError(f"missing duration for {node}")
        visiting.add(node)
        parents = dependencies.get(node, ())
        value = durations_ms[node] + max((finish(parent) for parent in parents), default=0)
        visiting.remove(node)
        cache[node] = value
        return value

    return max((finish(node) for node in durations_ms), default=0)


def tensor_elements(shape: Sequence[int]) -> int:
    """Return total elements after rejecting non-positive dimensions."""

    if not shape or min(shape) <= 0:
        raise ValueError("shape dimensions must be positive")
    return prod(shape)


def worked_example() -> dict[str, float | int]:
    """Return the shared Appendix B support-agent worksheet results."""

    cap = capacity(20, 3, 2, 0.4)
    path = critical_path_ms(
        {"classify": 180, "retrieve": 120, "policy": 60, "draft": 260},
        {"retrieve": ("classify",), "policy": ("classify",), "draft": ("retrieve", "policy")},
    )
    return {
        "lora_training_gib": training_memory_bytes(7_000_000_000, 35_000_000) / GIB,
        "training_zflop": training_flops(7_000_000_000, 100_000_000_000) / 1e21,
        "kv_gib": kv_cache_bytes(32, 8, 128, 32_768, 4) / GIB,
        "provisioned_inflight": cap.provisioned_inflight,
        "workers": cap.workers,
        "rag_gib": rag_storage_bytes(10_000_000, 1_024) / GIB,
        "media_gpu_seconds": media_gpu_seconds(1, candidates=4),
        "pass_pow_30": pass_pow_k(0.98, 30),
        "critical_path_ms": path,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(worked_example(), indent=2))
