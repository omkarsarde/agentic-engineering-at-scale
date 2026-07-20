"""Executable contracts for Chapter 6 distributed and frontier training."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CH06 = ROOT / "code" / "ch06"
sys.path.insert(0, str(CH06))

from cost_model import CostAssumptions, evaluate_cost, mfu_sensitivity  # noqa: E402
from memory_math import (  # noqa: E402
    ParallelPlan,
    adam_memory_ledger,
    expert_all_to_all_payload,
    pipeline_bubble_fraction,
    ring_allreduce_time,
)
from optimizer_probe import compare_optimizers, wsd_multiplier  # noqa: E402
from probe_model import GPTConfig, saved_tensor_bytes  # noqa: E402


_BUILD_SPEC = importlib.util.spec_from_file_location("ch06_run_build", CH06 / "run_build.py")
assert _BUILD_SPEC is not None and _BUILD_SPEC.loader is not None
_BUILD_MODULE = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(_BUILD_MODULE)
run_build = _BUILD_MODULE.run_build


def test_adam_16_byte_ledger_and_zero_stages() -> None:
    parameters, world = 1_000_000, 8
    rows = [adam_memory_ledger(parameters, world_size=world, stage=stage) for stage in range(4)]
    assert rows[0].persistent_bytes == 16 * parameters
    assert rows[1].optimizer_moments == 8 * parameters // world
    assert rows[2].gradients == 2 * parameters // world
    assert rows[3].parameters == 2 * parameters // world
    assert rows[3].persistent_bytes == 16 * parameters // world
    assert rows[3].communication_per_step > rows[2].communication_per_step


def test_parallelism_and_collective_algebra() -> None:
    assert pipeline_bubble_fraction(8, 32) == pytest.approx(7 / 39)
    assert ParallelPlan(data=2, tensor=4, pipeline=2, context=2, expert=2).devices == 32
    assert ParallelPlan(
        data=2, tensor=4, pipeline=2, context=2, expert=2, expert_is_independent=True
    ).devices == 64
    assert ring_allreduce_time(1_000_000, ranks=1, bandwidth_bytes_s=1e9, latency_s=1e-6) == 0
    multi = ring_allreduce_time(1_000_000, ranks=8, bandwidth_bytes_s=1e9, latency_s=1e-6)
    assert multi == pytest.approx(14e-6 + 1.75e-3)
    assert expert_all_to_all_payload(
        tokens=1_024, hidden_size=4_096, top_k=2, scalar_bytes=2, off_rank_fraction=0.75
    ) == 25_165_824


def test_cost_model_keeps_compute_fixed_and_prices_mfu() -> None:
    base = CostAssumptions()
    result = evaluate_cost(base)
    assert result["training_flops"] == 1.08e23
    assert result["accelerator_hours"] == pytest.approx(75_000)
    assert result["average_facility_mw"] == pytest.approx(9.8304)
    rows = mfu_sensitivity(base, (0.4, 0.5))
    assert rows[1]["training_cost"] / rows[0]["training_cost"] == pytest.approx(0.8)
    assert rows[1]["wall_days"] / rows[0]["wall_days"] == pytest.approx(0.8)


def test_checkpointing_reduces_autograd_saved_tensors() -> None:
    config = GPTConfig(vocab_size=128, block_size=32, d_model=32, n_heads=4, n_layers=2, mlp_ratio=2)
    without = saved_tensor_bytes(config, False, 7)
    with_checkpoint = saved_tensor_bytes(config, True, 7)
    assert with_checkpoint < without / 2


def test_wsd_and_muon_like_probe_are_finite_and_learn() -> None:
    assert wsd_multiplier(0, 20) < wsd_multiplier(5, 20) == 1
    assert wsd_multiplier(19, 20) == 0
    rows = compare_optimizers(steps=20)
    for optimizer in ("AdamW", "Muon-like"):
        selected = [row for row in rows if row["optimizer"] == optimizer]
        assert torch.isfinite(torch.tensor([row["training_loss"] for row in selected])).all()
        assert selected[-1]["training_loss"] < selected[0]["training_loss"]


def test_integrated_fsdp2_build_emits_measured_and_modeled_evidence(tmp_path: Path) -> None:
    metrics = run_build(tmp_path, world_size=2)
    probe = metrics["fsdp_probe"]
    assert probe["mode"] == "FSDP2 fully_shard"
    assert probe["world_size"] == 2
    assert {phase["activation_checkpointing"] for phase in probe["phases"]} == {False, True}
    assert probe["saved_tensor_bytes"]["with_checkpoint"] < probe["saved_tensor_bytes"]["without_checkpoint"]
    validation = metrics["memory_validation"]
    assert abs(validation["aggregate_relative_error"]) < 0.01
    predicted = validation["predicted_runtime_components_rank0"]
    measured = validation["measured_runtime_components_rank0"]
    assert measured["parameters"] == predicted["parameters"]
    assert measured["gradients"] == predicted["gradients"]
    assert validation["optimizer_excess_bytes_are_step_scalars"] > 0
    assert probe["matched_phase_loss_abs_delta"] < 1e-6
    assert probe["matched_phase_gradient_l2_abs_delta"] < 1e-6
    for filename in (
        "metrics.json",
        "fsdp-probe.json",
        "memory-ledger.csv",
        "runtime-memory-ledger.csv",
        "optimizer-curves.csv",
        "cost-sensitivity.csv",
        "memory-checkpoint.svg",
        "optimizer-curves.svg",
        "cost-sensitivity.svg",
        "cost-model.xlsx",
    ):
        assert (tmp_path / filename).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["cost_base_case"]["training_flops"] == 1.08e23
    with zipfile.ZipFile(tmp_path / "cost-model.xlsx") as archive:
        worksheets = [name for name in archive.namelist() if name.startswith("xl/worksheets/")]
        formulas = [
            element.text or ""
            for name in worksheets
            for element in ElementTree.fromstring(archive.read(name)).iter()
            if element.tag.endswith("}f")
        ]
    assert len(formulas) >= 12
    for sheet_name in ("Assumptions", "Model", "Sensitivity"):
        assert any(f"'{sheet_name}'!" in formula for formula in formulas)
        assert all(f"{sheet_name}!" not in formula for formula in formulas)
