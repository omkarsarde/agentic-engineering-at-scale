"""Small MoE and fixed-state mechanisms for Chapter 4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RoutingState:
    probabilities: Tensor
    weights: Tensor
    indices: Tensor
    load: Tensor
    balance_loss: Tensor
    z_loss: Tensor


class TopKRouter(nn.Module):
    """Route each token to k experts and expose balancing diagnostics."""

    def __init__(self, width: int, experts: int, top_k: int, scoring: str = "softmax") -> None:
        super().__init__()
        if not 1 <= top_k <= experts or scoring not in {"softmax", "sigmoid"}:
            raise ValueError("invalid top_k or scoring function")
        self.experts, self.top_k, self.scoring = experts, top_k, scoring
        self.projection = nn.Linear(width, experts)
        self.register_buffer("selection_bias", torch.zeros(experts))

    def forward(self, tokens: Tensor) -> RoutingState:
        logits = self.projection(tokens)
        probabilities = logits.softmax(-1) if self.scoring == "softmax" else logits.sigmoid()
        indices = (probabilities + self.selection_bias).topk(self.top_k, dim=-1).indices
        weights = probabilities.gather(-1, indices)
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        flat_indices = indices.reshape(-1, self.top_k)
        load = F.one_hot(flat_indices, self.experts).float().sum(1).mean(0) / self.top_k
        mean_probability = probabilities.reshape(-1, self.experts).mean(0)
        balance_loss = self.experts * (load.detach() * mean_probability).sum()
        z_loss = logits.logsumexp(-1).square().mean()
        return RoutingState(probabilities, weights, indices, load, balance_loss, z_loss)

    @torch.no_grad()
    def update_selection_bias(self, load: Tensor, rate: float = 1e-2) -> None:
        """Apply the sign update used by an aux-loss-free balancing controller."""

        target = torch.full_like(load, 1 / self.experts)
        self.selection_bias.add_(rate * torch.sign(target - load))
        self.selection_bias.sub_(self.selection_bias.mean())


class ToyMoE(nn.Module):
    """A local dispatch implementation; no expert-parallel communication."""

    def __init__(self, width: int = 8, experts: int = 4, hidden: int = 16) -> None:
        super().__init__()
        self.router = TopKRouter(width, experts, top_k=1)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 4))
            for _ in range(experts)
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, RoutingState]:
        routing = self.router(tokens)
        flat = tokens.reshape(-1, tokens.size(-1))
        indices = routing.indices.reshape(-1, 1)
        weights = routing.weights.reshape(-1, 1)
        output = flat.new_zeros(flat.size(0), 4)
        for slot in range(routing.indices.size(-1)):
            for expert_id, expert in enumerate(self.experts):
                rows = torch.where(indices[:, slot] == expert_id)[0]
                if rows.numel():
                    output[rows] += weights[rows, slot, None] * expert(flat[rows])
        return output.view(*tokens.shape[:-1], 4), routing


def train_router(balance_weight: float, steps: int = 300) -> dict[str, object]:
    """Train one seeded classification MoE with or without load balancing."""

    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = ToyMoE()
    nn.init.zeros_(model.router.projection.weight)
    nn.init.constant_(model.router.projection.bias, -1.0)
    model.router.projection.bias.data[0] = 2.0
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    generator = torch.Generator().manual_seed(8)
    for _ in range(steps):
        tokens = torch.randn(256, 8, generator=generator)
        labels = tokens[:, :4].argmax(-1)
        logits, routing = model(tokens)
        loss = (
            F.cross_entropy(logits, labels)
            + balance_weight * routing.balance_loss
            + 1e-4 * routing.z_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        tokens = torch.randn(8_192, 8, generator=generator)
        labels = tokens[:, :4].argmax(-1)
        logits, routing = model(tokens)
    return {
        "balance_weight": balance_weight,
        "load": routing.load.tolist(),
        "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        "balance_loss": routing.balance_loss.item(),
        "z_loss": routing.z_loss.item(),
    }


class GatedLinearAttention(nn.Module):
    """A recurrent kernel-attention block with context-independent state size."""

    def __init__(self, width: int, state_width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, state_width, bias=False)
        self.key = nn.Linear(width, state_width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, 1)
        self.output = nn.Linear(width, width, bias=False)

    def forward(
        self, tokens: Tensor, recurrent: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, steps, width = tokens.shape
        state_width = self.query.out_features
        if recurrent is None:
            state = tokens.new_zeros(batch, state_width, width)
            normalizer = tokens.new_zeros(batch, state_width)
        else:
            state, normalizer = recurrent
        outputs = []
        for step in range(steps):
            token = tokens[:, step]
            query = F.elu(self.query(token)) + 1
            key = F.elu(self.key(token)) + 1
            value = self.value(token)
            decay = self.decay(token).sigmoid()
            state = decay[:, :, None] * state + torch.einsum("bf,bd->bfd", key, value)
            normalizer = decay * normalizer + key
            numerator = torch.einsum("bf,bfd->bd", query, state)
            denominator = (query * normalizer).sum(-1, keepdim=True).clamp_min(1e-6)
            outputs.append(self.output(numerator / denominator))
        return torch.stack(outputs, dim=1), (state, normalizer)


def mqar_capacity_curve(
    pair_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    state_slots: tuple[int, ...] = (8, 32),
    trials: int = 4_096,
) -> list[dict[str, float | int | str]]:
    """Compare exact lookup with bounded hashed feature-state recall."""

    rows: list[dict[str, float | int | str]] = []
    for pairs in pair_counts:
        rows.append({"memory": "full attention", "pairs": pairs, "accuracy": 1.0})
        for slots in state_slots:
            generator = torch.Generator().manual_seed(10_000 + pairs * 100 + slots)
            buckets = torch.randint(slots, (trials, pairs), generator=generator)
            query = torch.randint(pairs, (trials,), generator=generator)
            query_bucket = buckets.gather(1, query[:, None])
            positions = torch.arange(pairs).expand(trials, -1)
            last_in_bucket = positions.masked_fill(buckets != query_bucket, -1).max(1).values
            accuracy = (last_in_bucket == query).float().mean().item()
            rows.append({"memory": f"fixed state ({slots} slots)", "pairs": pairs, "accuracy": accuracy})
    return rows
