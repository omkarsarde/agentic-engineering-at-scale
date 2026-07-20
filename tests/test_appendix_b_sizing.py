"""Focused unit and boundary tests for Appendix B sizing worksheets."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "appb"
sys.path.insert(0, str(CODE))

from sizing import (  # noqa: E402
    GIB,
    capacity,
    critical_path_ms,
    kv_cache_bytes,
    media_gpu_seconds,
    pass_pow_k,
    rag_storage_bytes,
    training_memory_bytes,
    worked_example,
)


def test_frozen_parameters_do_not_pay_optimizer_levy() -> None:
    full = training_memory_bytes(100, 100)
    lora = training_memory_bytes(100, 10)
    assert full == 1600
    assert lora == 340


def test_kv_cache_matches_explicit_shape_product() -> None:
    actual = kv_cache_bytes(32, 8, 128, 32_768, batch=4, dtype_bytes=2)
    assert actual == 2 * 32 * 8 * 128 * 32_768 * 4 * 2
    assert actual / GIB == 16


def test_capacity_applies_fanout_and_headroom() -> None:
    estimate = capacity(20, 3, 2, 0.4, headroom=1.3)
    assert estimate.mean_inflight == 120
    assert estimate.provisioned_inflight == 156
    assert estimate.workers == 32


def test_storage_and_media_units_are_explicit() -> None:
    assert rag_storage_bytes(10, 4, dtype_bytes=2, index_overhead=1.5, replicas=2) == 240
    assert media_gpu_seconds(1, video_seconds=8, candidates=2) == 88


def test_long_trajectory_reliability_decays() -> None:
    assert round(pass_pow_k(0.98, 30), 3) == 0.545
    assert pass_pow_k(0.98, 30) < 0.98


def test_critical_path_uses_parallel_max_not_sum() -> None:
    duration = critical_path_ms(
        {"a": 100, "b": 200, "c": 80, "d": 50},
        {"b": ("a",), "c": ("a",), "d": ("b", "c")},
    )
    assert duration == 350


def test_cycle_is_rejected() -> None:
    try:
        critical_path_ms({"a": 1, "b": 1}, {"a": ("b",), "b": ("a",)})
    except ValueError:
        pass
    else:
        raise AssertionError("cyclic dependency graph was accepted")


def test_worked_example_has_every_worksheet() -> None:
    report = worked_example()
    assert set(report) == {
        "lora_training_gib",
        "training_zflop",
        "kv_gib",
        "provisioned_inflight",
        "workers",
        "rag_gib",
        "media_gpu_seconds",
        "pass_pow_30",
        "critical_path_ms",
    }
