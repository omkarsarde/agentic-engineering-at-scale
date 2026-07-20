"""Request-phase roofline and batch-order probes for Chapter 9."""

from __future__ import annotations

from decoding_runtime import softmax


MODEL = {"parameters": 7_000_000_000, "layers": 32, "kv_heads": 8, "head_dim": 128}
HARDWARE = {"peak_flops": 80e12, "memory_bandwidth": 2e12, "dtype_bytes": 2}


def kv_bytes(tokens: int, batch: int = 1) -> int:
    """Compute decoder KV bytes for the illustrative grouped-query model."""
    return (2 * MODEL["layers"] * MODEL["kv_heads"] * MODEL["head_dim"]
            * tokens * batch * HARDWARE["dtype_bytes"])


def request_profiles(output_tokens: int = 128) -> list[dict[str, float | int | str]]:
    """Place illustrative prefill and decode work on a simple roofline."""
    parameter_bytes = MODEL["parameters"] * HARDWARE["dtype_bytes"]
    ridge = HARDWARE["peak_flops"] / HARDWARE["memory_bandwidth"]
    rows = []
    for tokens in (128, 512, 2048):
        prefill_ms = max(2 * MODEL["parameters"] * tokens / HARDWARE["peak_flops"],
                         parameter_bytes / HARDWARE["memory_bandwidth"]) * 1000
        decode_bytes = parameter_bytes + kv_bytes(tokens)
        tpot_ms = max(2 * MODEL["parameters"] / HARDWARE["peak_flops"],
                      decode_bytes / HARDWARE["memory_bandwidth"]) * 1000
        ttft_ms = prefill_ms + tpot_ms
        rows.append({"prompt_tokens": tokens, "kv_mib": kv_bytes(tokens) / 2**20,
                     "prefill_intensity": float(tokens), "decode_intensity": 1.0,
                     "ridge_flops_per_byte": ridge, "prefill_bound": "compute" if tokens > ridge else "bandwidth",
                     "decode_bound": "bandwidth", "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
                     "total_ms": ttft_ms + (output_tokens - 1) * tpot_ms})
    return rows


def batch_invariance_probe() -> dict[str, float | int]:
    """Expose a greedy flip caused only by two legal reduction orders."""
    parts = [1e16, 1.0, -1e16, 0.0]
    # Spell out the reduction order. Python 3.12's built-in ``sum`` uses a
    # more accurate algorithm than older interpreters, which would erase the
    # floating-point-ordering lesson this deterministic probe is built to show.
    serial = 0.0
    for part in parts:
        serial += part
    partitioned = (parts[0] + parts[2]) + (parts[1] + parts[3])
    single = softmax([serial, 0.5])
    batched = softmax([partitioned, 0.5])
    return {"single_token": max(range(2), key=single.__getitem__),
            "batched_token": max(range(2), key=batched.__getitem__),
            "greedy_flips": int(single.index(max(single)) != batched.index(max(batched))),
            "max_probability_drift": max(abs(a - b) for a, b in zip(single, batched))}
