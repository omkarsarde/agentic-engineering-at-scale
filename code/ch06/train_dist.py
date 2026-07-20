"""A small FSDP2 training probe with activation-checkpoint and MFU instrumentation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from instrumentation import local_bytes, local_squared_sum, matmul_roofline
from probe_model import GPTConfig, TrainingTinyGPT, saved_tensor_bytes


def distributed_probe(
    output: Path,
    *,
    steps: int = 4,
    seed: int = 73,
    declared_peak_tflops: float = 0.0,
    precision: str = "fp32",
) -> dict[str, object] | None:
    """Run two measured FSDP2 phases and write rank-zero JSON evidence."""

    if steps < 2 or declared_peak_tflops < 0 or precision not in {"fp32", "bf16"}:
        raise ValueError("use at least two steps, nonnegative peak throughput, and fp32 or bf16")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = GPTConfig(vocab_size=320, block_size=64, d_model=64, n_heads=4, n_layers=2, mlp_ratio=2)
    saved = None
    if rank == 0:
        saved = {
            "without_checkpoint": saved_tensor_bytes(config, False, seed),
            "with_checkpoint": saved_tensor_bytes(config, True, seed),
        }
    mesh = init_device_mesh(device.type, (world,))
    mp_policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if precision == "bf16"
        else MixedPrecisionPolicy()
    )
    roofline = torch.tensor(matmul_roofline(device), device=device)
    dist.all_reduce(roofline, op=dist.ReduceOp.SUM)
    phases = []
    local_state: dict[str, int] = {}
    parameter_count = largest_unit = 0
    unit_counts: list[int] = []
    for checkpoint_blocks in (False, True):
        # Both conditions restart from identical weights, Adam state, and rank-local batches.
        torch.manual_seed(seed)
        model = TrainingTinyGPT(config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        block_counts = [sum(p.numel() for p in block.parameters()) for block in model.base.blocks]
        root_owned = parameter_count - sum(block_counts)
        unit_counts = [*block_counts, root_owned]
        largest_unit = max(unit_counts)
        for block in model.base.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=mesh, reshard_after_forward=False, mp_policy=mp_policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        generator = torch.Generator().manual_seed(seed + rank)
        batches = [
            torch.randint(config.vocab_size, (4, config.block_size), generator=generator)
            for _ in range(steps + 1)
        ]
        warmup_tokens = batches[0].to(device)
        optimizer.zero_grad(set_to_none=True)
        model(warmup_tokens, torch.roll(warmup_tokens, -1, dims=1), checkpoint_blocks).backward()
        optimizer.step()
        dist.barrier()
        started = time.perf_counter()
        last_loss = torch.tensor(0.0, device=device)
        for cpu_tokens in batches[1:]:
            tokens = cpu_tokens.to(device)
            targets = torch.roll(tokens, -1, dims=1)
            optimizer.zero_grad(set_to_none=True)
            last_loss = model(tokens, targets, checkpoint_blocks)
            last_loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier()
        elapsed = time.perf_counter() - started
        gradient_squared_sum = torch.tensor(
            sum(local_squared_sum(parameter.grad) for parameter in model.parameters()),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(gradient_squared_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(last_loss, op=dist.ReduceOp.SUM)
        training_tokens = steps * 4 * config.block_size * world
        model_flops = 6 * parameter_count * training_tokens
        capacity = (
            declared_peak_tflops * 1e12 * world
            if declared_peak_tflops
            else float(roofline)
        )
        phases.append(
            {
                "activation_checkpointing": checkpoint_blocks,
                "elapsed_seconds": elapsed,
                "tokens_per_second": training_tokens / elapsed,
                "last_loss": float((last_loss / world).detach()),
                "gradient_l2": float(gradient_squared_sum.sqrt()),
                "model_flops": model_flops,
                "mfu": model_flops / elapsed / capacity,
            }
        )
        local_state = {
            "parameters": sum(local_bytes(parameter) for parameter in model.parameters()),
            "gradients": sum(local_bytes(parameter.grad) for parameter in model.parameters()),
            "optimizer": sum(
                local_bytes(value)
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ),
        }
    result = None
    if rank == 0:
        result = {
            "mode": "FSDP2 fully_shard",
            "precision_policy": precision,
            "device": device.type,
            "world_size": world,
            "parameter_count": parameter_count,
            "largest_unit_parameters": largest_unit,
            "unit_parameter_counts": unit_counts,
            "config": config.__dict__,
            "saved_tensor_bytes": saved,
            "local_state_bytes_rank0": local_state,
            "mfu_denominator": (
                f"declared {declared_peak_tflops} TFLOP/s per rank"
                if declared_peak_tflops
                else "aggregate local matmul roofline; CPU/GPU proxy, not vendor peak"
            ),
            "aggregate_reference_flops_s": capacity,
            "phases": phases,
            "matched_phase_loss_abs_delta": abs(
                float(phases[0]["last_loss"]) - float(phases[1]["last_loss"])
            ),
            "matched_phase_gradient_l2_abs_delta": abs(
                float(phases[0]["gradient_l2"]) - float(phases[1]["gradient_l2"])
            ),
            "seed": seed,
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--declared-peak-tflops", type=float, default=0.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args()
    distributed_probe(
        args.output,
        steps=args.steps,
        declared_peak_tflops=args.declared_peak_tflops,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
