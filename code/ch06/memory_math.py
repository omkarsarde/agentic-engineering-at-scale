"""Memory and communication ledgers for distributed Transformer training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemoryLedger:
    """Steady-state per-rank bytes plus one transient FSDP unit."""

    strategy: str
    parameters: int
    gradients: int
    master_weights: int
    optimizer_moments: int
    activations: int
    transient_full_parameters: int
    communication_per_step: int

    @property
    def persistent_bytes(self) -> int:
        """Return model-state and activation bytes resident between collectives."""

        return (
            self.parameters
            + self.gradients
            + self.master_weights
            + self.optimizer_moments
            + self.activations
        )

    @property
    def peak_estimate_bytes(self) -> int:
        """Return persistent bytes plus the largest unsharded parameter unit."""

        return self.persistent_bytes + self.transient_full_parameters

    def as_dict(self) -> dict[str, int | str]:
        """Return a flat serializable record including totals."""

        return {
            **asdict(self),
            "persistent_bytes": self.persistent_bytes,
            "peak_estimate_bytes": self.peak_estimate_bytes,
        }


def adam_memory_ledger(
    parameter_count: int,
    *,
    world_size: int,
    stage: int,
    activation_bytes: int = 0,
    largest_unit_parameters: int = 0,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    master_weight_bytes: int = 4,
    moment_bytes: int = 8,
) -> MemoryLedger:
    """Account mixed-precision Adam state for DDP or ZeRO stages 1--3.

    Stage 0 is ordinary replicated DDP.  Stage 1 shards optimizer moments and
    master weights; stage 2 also shards gradients; stage 3/FSDP also shards
    parameters outside computation.  Communication is a ring-equivalent payload
    model, not a latency or topology prediction.
    """

    if parameter_count < 1 or world_size < 1 or stage not in range(4):
        raise ValueError("positive sizes and ZeRO stage 0, 1, 2, or 3 are required")
    shard = lambda value, enabled: (value + world_size - 1) // world_size if enabled else value
    parameters = shard(parameter_count * parameter_bytes, stage >= 3)
    gradients = shard(parameter_count * gradient_bytes, stage >= 2)
    master = shard(parameter_count * master_weight_bytes, stage >= 1)
    moments = shard(parameter_count * moment_bytes, stage >= 1)
    ring_fraction = (world_size - 1) / world_size
    if stage < 2:
        communication = 2 * parameter_count * gradient_bytes * ring_fraction
    elif stage == 2:
        communication = parameter_count * (gradient_bytes + parameter_bytes) * ring_fraction
    else:
        communication = parameter_count * (2 * parameter_bytes + gradient_bytes) * ring_fraction
    return MemoryLedger(
        strategy=("DDP", "ZeRO-1", "ZeRO-2", "FSDP2 / ZeRO-3")[stage],
        parameters=parameters,
        gradients=gradients,
        master_weights=master,
        optimizer_moments=moments,
        activations=activation_bytes,
        transient_full_parameters=(largest_unit_parameters * parameter_bytes if stage == 3 else 0),
        communication_per_step=round(communication),
    )


def pipeline_bubble_fraction(stages: int, microbatches: int) -> float:
    """Return the idealized 1F1B fill/drain bubble fraction."""

    if stages < 1 or microbatches < 1:
        raise ValueError("stages and microbatches must be positive")
    return (stages - 1) / (microbatches + stages - 1)


@dataclass(frozen=True)
class ParallelPlan:
    """Independent logical mesh dimensions for a training layout."""

    data: int = 1
    tensor: int = 1
    pipeline: int = 1
    context: int = 1
    expert: int = 1
    expert_is_independent: bool = False

    @property
    def devices(self) -> int:
        """Return the product when every dimension is physically independent."""

        values = (self.data, self.tensor, self.pipeline, self.context, self.expert)
        if any(value < 1 for value in values):
            raise ValueError("parallel dimensions must be positive")
        product = 1
        for value in (self.data, self.tensor, self.pipeline, self.context):
            product *= value
        if self.expert_is_independent:
            product *= self.expert
        return product

    def as_dict(self) -> dict[str, int | bool]:
        """Return mesh dimensions and their simple product."""

        return {**asdict(self), "devices": self.devices}


def ring_allreduce_time(
    payload_bytes: int, *, ranks: int, bandwidth_bytes_s: float, latency_s: float
) -> float:
    """Estimate ring all-reduce time from payload, link bandwidth, and latency."""

    if payload_bytes < 0 or ranks < 1 or bandwidth_bytes_s <= 0 or latency_s < 0:
        raise ValueError("collective inputs are outside their physical domains")
    if ranks == 1:
        return 0.0
    return 2 * (ranks - 1) * latency_s + 2 * (ranks - 1) / ranks * payload_bytes / bandwidth_bytes_s


def expert_all_to_all_payload(
    *, tokens: int, hidden_size: int, top_k: int, scalar_bytes: int, off_rank_fraction: float
) -> int:
    """Return dispatch-plus-combine bytes for off-rank routed token states."""

    if min(tokens, hidden_size, top_k, scalar_bytes) < 1 or not 0 <= off_rank_fraction <= 1:
        raise ValueError("expert payload inputs are outside their physical domains")
    return round(2 * tokens * hidden_size * top_k * scalar_bytes * off_rank_fraction)
