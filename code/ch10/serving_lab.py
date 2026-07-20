"""Deterministic CPU lab for scheduling, caching, and serving decisions."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass


REQUEST_LENGTHS = [2, 12, 4, 10, 3, 9, 5, 8, 6, 7, 2, 11]


@dataclass(frozen=True)
class CacheKey:
    """Identity fields that make a cached prefix safe to reuse."""

    tenant: str
    model_revision: str
    tokenizer_revision: str
    template_revision: str
    adapter_revision: str
    prefix_hash: str


def static_schedule(lengths: list[int], batch_size: int) -> dict[str, object]:
    """Run fixed batches that wait for their longest member."""
    starts, finishes, elapsed = [0] * len(lengths), [0] * len(lengths), 0
    for base in range(0, len(lengths), batch_size):
        group = lengths[base:base + batch_size]
        for offset, length in enumerate(group):
            starts[base + offset] = elapsed
            finishes[base + offset] = elapsed + length
        elapsed += max(group)
    capacity = batch_size * elapsed
    return {"starts": starts, "finishes": finishes, "makespan_steps": elapsed,
            "utilization": sum(lengths) / capacity}


def continuous_schedule(lengths: list[int], batch_size: int) -> dict[str, object]:
    """Refill a decode slot as soon as its current request completes."""
    starts, finishes = [-1] * len(lengths), [-1] * len(lengths)
    remaining, active, waiting, step = lengths[:], [], list(range(len(lengths))), 0
    while waiting and len(active) < batch_size:
        request = waiting.pop(0); starts[request] = step; active.append(request)
    used_slots = 0
    while active:
        used_slots += len(active)
        for request in active:
            remaining[request] -= 1
        step += 1
        completed = [request for request in active if remaining[request] == 0]
        for request in completed:
            finishes[request] = step; active.remove(request)
        while waiting and len(active) < batch_size:
            request = waiting.pop(0); starts[request] = step; active.append(request)
    return {"starts": starts, "finishes": finishes, "makespan_steps": step,
            "utilization": used_slots / (batch_size * step)}


def goodput(schedule: dict[str, object], lengths: list[int], ttft_slo: int = 16,
            tpot_slo: float = 1.05) -> dict[str, float | int]:
    """Count requests satisfying both synthetic latency objectives per step."""
    starts, finishes = list(schedule["starts"]), list(schedule["finishes"])
    qualified = sum(start + 1 <= ttft_slo and (finish - start) / length <= tpot_slo
                    for start, finish, length in zip(starts, finishes, lengths))
    return {"qualified_requests": qualified,
            "requests_per_step": qualified / int(schedule["makespan_steps"])}


def make_cache_key(tenant: str, token_ids: list[int], *, model: str = "model-r1",
                   tokenizer: str = "tok-r1", template: str = "tmpl-r1",
                   adapter: str = "base") -> CacheKey:
    """Hash exact token IDs together with every state-changing revision."""
    digest = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    return CacheKey(tenant, model, tokenizer, template, adapter, digest)


def prefix_cache_probe(tokens: int = 45, block_size: int = 16) -> dict[str, object]:
    """Measure full-block sharing, tail waste, and tenant isolation."""
    full, tail = divmod(tokens, block_size)
    token_ids = list(range(tokens))
    a = make_cache_key("tenant-a", token_ids)
    a_again = make_cache_key("tenant-a", token_ids)
    b = make_cache_key("tenant-b", token_ids)
    return {"full_blocks_shared_per_branch": full, "private_tail_tokens": tail,
            "tail_capacity_wasted_per_branch": (block_size - tail) % block_size,
            "blocks_without_sharing_two_branches": 2 * (full + bool(tail)),
            "blocks_with_safe_sharing_two_branches": full + 2 * bool(tail),
            "same_identity_reuses": a == a_again, "cross_tenant_reuses": a == b}


def cache_breakeven(compute: float, write: float, read: float) -> float:
    """Return future reuses needed for a cache write to beat recomputation."""
    if read >= compute:
        return math.inf
    return max(0.0, (write - compute) / (compute - read))


def categorical(probs: list[float], rng: random.Random) -> int:
    """Draw a categorical index from the supplied random generator."""
    draw, total = rng.random(), 0.0
    for index, probability in enumerate(probs):
        total += probability
        if draw <= total:
            return index
    return len(probs) - 1


def speculative_draw(target: list[float], draft: list[float], rng: random.Random) -> tuple[int, bool]:
    """Sample one exact target token through speculative accept/reject."""
    candidate = categorical(draft, rng)
    if rng.random() <= min(1.0, target[candidate] / draft[candidate]):
        return candidate, True
    residual = [max(p - q, 0.0) for p, q in zip(target, draft)]
    mass = sum(residual)
    return categorical([value / mass for value in residual], rng), False


def speculation_probe(draws: int = 40_000) -> dict[str, float]:
    """Verify target preservation and compare two draft-cost regimes."""
    target, draft = [0.52, 0.28, 0.15, 0.05], [0.42, 0.32, 0.18, 0.08]
    counts, accepted = [0] * len(target), 0
    rng = random.Random(731)
    for _ in range(draws):
        token, was_accepted = speculative_draw(target, draft, rng)
        counts[token] += 1; accepted += was_accepted
    empirical = [count / draws for count in counts]
    acceptance = sum(min(p, q) for p, q in zip(target, draft))
    expected_emitted = sum(acceptance ** position for position in range(5))
    return {"acceptance_probability": acceptance,
            "empirical_acceptance": accepted / draws,
            "target_total_variation": sum(abs(a - b) for a, b in zip(empirical, target)) / 2,
            "speedup_low_load": expected_emitted / (1 + 4 * 0.05),
            "speedup_saturated": expected_emitted / (1 + 4 * 0.85)}


def constrained_distribution(probs: list[float], allowed: set[int]) -> dict[str, object]:
    """Mask invalid next tokens and report the probability mass removed."""
    kept = sum(probs[index] for index in allowed)
    if kept == 0:
        raise ValueError("grammar admits no token")
    result = [probability / kept if index in allowed else 0.0
              for index, probability in enumerate(probs)]
    return {"distribution": result, "removed_mass": 1 - kept}


def quantize_blockwise(values: list[float], bits: int, block_size: int) -> tuple[list[float], float]:
    """Apply symmetric integer quantization independently to each block."""
    qmax, restored = 2 ** (bits - 1) - 1, []
    for base in range(0, len(values), block_size):
        block = values[base:base + block_size]
        scale = max(abs(value) for value in block) / qmax or 1.0
        restored.extend(max(-qmax, min(qmax, round(value / scale))) * scale for value in block)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values))
    return restored, rmse


def quantization_probe() -> list[dict[str, float | int]]:
    """Compare four-bit error and metadata overhead across block sizes."""
    values = [2 * math.sin(i / 11) + 0.3 * math.cos(i / 3) for i in range(256)]
    values[47], values[181] = 38.0, -29.0
    rows = []
    for block_size in (16, 64, 256):
        _, rmse = quantize_blockwise(values, 4, block_size)
        rows.append({"block_size": block_size, "rmse": rmse,
                     "effective_bits_per_value": 4 + 16 / block_size})
    return rows


def disaggregation_probe() -> list[dict[str, float | bool | str]]:
    """Apply the transfer-versus-interference inequality to two fixtures."""
    rows = []
    for name, kv_gib, saved_ms in (("short prompt", 0.25, 3.0), ("long prompt", 4.0, 35.0)):
        transfer_ms = kv_gib / 200 * 1000
        overhead_ms = transfer_ms + 0.8 + 2.0
        rows.append({"workload": name, "kv_gib": kv_gib, "interference_saved_ms": saved_ms,
                     "transfer_and_coordination_ms": overhead_ms,
                     "disaggregate": saved_ms > overhead_ms})
    return rows


def reasoning_budget_probe() -> list[dict[str, float | int]]:
    """Generate a synthetic saturating quality curve with an overthinking tail."""
    rows = []
    for tokens in (32, 64, 128, 256, 512):
        score = 0.58 + 0.22 * (1 - math.exp(-tokens / 120)) - 0.00020 * max(0, tokens - 384)
        rows.append({"reasoning_tokens": tokens, "synthetic_score": score,
                     "score_per_1k_tokens": score * 1000 / tokens})
    return rows


def run_experiment() -> dict[str, object]:
    """Run every deterministic serving probe and return one measurement record."""
    static, continuous = static_schedule(REQUEST_LENGTHS, 4), continuous_schedule(REQUEST_LENGTHS, 4)
    return {"fixture": {"request_lengths": REQUEST_LENGTHS, "batch_size": 4,
                        "ttft_slo_steps": 16, "tpot_slo_steps": 1.05},
            "scheduling": {"static": static, "continuous": continuous,
                           "static_goodput": goodput(static, REQUEST_LENGTHS),
                           "continuous_goodput": goodput(continuous, REQUEST_LENGTHS)},
            "prefix_cache": prefix_cache_probe(),
            "cache_breakeven_reuses": cache_breakeven(1.0, 1.25, 0.10),
            "speculation": speculation_probe(),
            "constraint": constrained_distribution([0.60, 0.25, 0.15], {1, 2}),
            "quantization": quantization_probe(),
            "disaggregation": disaggregation_probe(),
            "reasoning_budget": reasoning_budget_probe()}


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
