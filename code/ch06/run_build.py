"""Run the integrated Chapter 6 FSDP, optimizer, memory, and economics build."""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from cost_model import CostAssumptions, evaluate_cost, mfu_sensitivity
from memory_math import adam_memory_ledger
from optimizer_probe import compare_optimizers
from render import render_figures


HERE = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def launch_fsdp_probe(
    output: Path, *, world_size: int | None = None, steps: int = 4
) -> dict[str, object]:
    """Launch the real FSDP2 probe under ``torch.distributed.run``."""

    if world_size is None:
        world_size = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 2
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if torch.cuda.is_available() and world_size > torch.cuda.device_count():
        raise ValueError("world_size cannot exceed the visible CUDA device count")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(HERE / "train_dist.py"),
        "--output",
        str(output),
        "--steps",
        str(steps),
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180)
    if completed.returncode:
        raise RuntimeError(f"FSDP2 probe failed:\n{completed.stderr[-4_000:]}")
    return json.loads(output.read_text(encoding="utf-8"))


def run_build(
    out_dir: Path = HERE / "generated", *, world_size: int | None = None
) -> dict[str, object]:
    """Compose one seeded laboratory with explicitly non-deterministic timing fields."""

    out_dir.mkdir(parents=True, exist_ok=True)
    probe = launch_fsdp_probe(out_dir / "fsdp-probe.json", world_size=world_size)
    world_size = int(probe["world_size"])
    parameters = int(probe["parameter_count"])
    checkpoint_bytes = int(probe["saved_tensor_bytes"]["with_checkpoint"])
    memory_rows = [
        adam_memory_ledger(
            parameters,
            world_size=world_size,
            stage=stage,
            activation_bytes=checkpoint_bytes,
            largest_unit_parameters=int(probe["largest_unit_parameters"]),
        ).as_dict()
        for stage in range(4)
    ]
    runtime_memory_rows = [
        adam_memory_ledger(
            parameters,
            world_size=world_size,
            stage=stage,
            activation_bytes=checkpoint_bytes,
            largest_unit_parameters=int(probe["largest_unit_parameters"]),
            parameter_bytes=4,
            gradient_bytes=4,
            master_weight_bytes=0,
            moment_bytes=8,
        ).as_dict()
        for stage in range(4)
    ]
    optimizer_rows = compare_optimizers()
    assumptions = CostAssumptions()
    cost_rows = mfu_sensitivity(assumptions)
    measured_components = {
        "parameters": int(probe["local_state_bytes_rank0"]["parameters"]),
        "gradients": int(probe["local_state_bytes_rank0"]["gradients"]),
        "optimizer_moments_plus_step_scalars": int(
            probe["local_state_bytes_rank0"]["optimizer"]
        ),
    }
    predicted_components = {
        "parameters": int(runtime_memory_rows[-1]["parameters"]),
        "gradients": int(runtime_memory_rows[-1]["gradients"]),
        "optimizer_moments": int(runtime_memory_rows[-1]["optimizer_moments"]),
    }
    actual_state = sum(measured_components.values())
    predicted_state = sum(predicted_components.values())
    optimizer_summary = {
        optimizer: {
            "initial_loss": next(float(row["training_loss"]) for row in optimizer_rows if row["optimizer"] == optimizer),
            "final_loss": next(float(row["training_loss"]) for row in reversed(optimizer_rows) if row["optimizer"] == optimizer),
        }
        for optimizer in ("AdamW", "Muon-like")
    }
    metrics: dict[str, object] = {
        "experiment_contracts": {
            "fsdp": "real FSDP2/DTensor collectives; CPU gloo by default, accelerator-compatible",
            "mfu": probe["mfu_denominator"],
            "cost": "illustrative assumptions, not a vendor quote or procurement forecast",
            "optimizer": "single seeded TinyGPT mechanism probe, not a speedrun ranking",
        },
        "fsdp_probe": probe,
        "memory_validation": {
            "runtime_layout": "FP32 parameters + FP32 gradients + FP32 Adam moments; no master copy",
            "classic_baseline_layout": "BF16 parameters + BF16 gradients + FP32 master + FP32 Adam moments",
            "predicted_runtime_components_rank0": predicted_components,
            "measured_runtime_components_rank0": measured_components,
            "predicted_runtime_total_rank0": predicted_state,
            "measured_runtime_total_rank0": actual_state,
            "aggregate_relative_error": (actual_state - predicted_state) / predicted_state,
            "optimizer_excess_bytes_are_step_scalars": (
                measured_components["optimizer_moments_plus_step_scalars"]
                - predicted_components["optimizer_moments"]
            ),
        },
        "optimizer_summary": optimizer_summary,
        "cost_base_case": evaluate_cost(assumptions),
        "reproducibility": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "world_size": world_size,
            "seed_fsdp": probe["seed"],
            "seed_optimizer": 61,
        },
    }
    _write_csv(out_dir / "memory-ledger.csv", memory_rows)
    _write_csv(out_dir / "runtime-memory-ledger.csv", runtime_memory_rows)
    _write_csv(out_dir / "optimizer-curves.csv", optimizer_rows)
    _write_csv(out_dir / "cost-sensitivity.csv", cost_rows)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_figures(memory_rows, probe, optimizer_rows, cost_rows, out_dir)
    shutil.copyfile(HERE / "fixtures" / "cost-model.xlsx", out_dir / "cost-model.xlsx")
    return metrics


if __name__ == "__main__":
    run_build()
