"""Canonical KV-cache arithmetic and small attention mechanisms for Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rope import apply_rope, inverse_frequencies


@dataclass(frozen=True)
class KVConfig:
    """The architectural quantities that determine persistent KV state."""

    name: str
    layers: int
    query_heads: int
    head_dim: int
    bytes_per_scalar: int
    kv_heads: int | None = None
    latent_rank: int | None = None
    rope_key_dim: int = 0

    def cached_scalars_per_layer_token(self) -> int:
        if self.latent_rank is not None:
            return self.latent_rank + self.rope_key_dim
        if self.kv_heads is None:
            raise ValueError("standard attention requires kv_heads")
        return 2 * self.kv_heads * self.head_dim


def kv_bytes(config: KVConfig, tokens: int, batch: int = 1) -> int:
    """Return payload bytes for one append-only, unquantized KV cache.

    This is the book's canonical declaration. It excludes allocator metadata,
    padding, temporary attention workspaces, and tensor-parallel duplication.
    """

    if tokens < 0 or batch < 1:
        raise ValueError("tokens must be nonnegative and batch must be positive")
    return (
        batch
        * tokens
        * config.layers
        * config.cached_scalars_per_layer_token()
        * config.bytes_per_scalar
    )


class GroupedQueryAttention(nn.Module):
    """One implementation whose kv-head dial spans MHA, GQA, and MQA."""

    def __init__(self, d_model: int, query_heads: int, kv_heads: int) -> None:
        super().__init__()
        if d_model % query_heads or query_heads % kv_heads:
            raise ValueError("query heads must divide width and be divisible by kv heads")
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = d_model // query_heads
        self.q_proj = nn.Linear(d_model, query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: Tensor, cache: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, query_len, _ = x.shape
        q = self.q_proj(x).view(batch, query_len, self.query_heads, self.head_dim)
        k = self.k_proj(x).view(batch, query_len, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, query_len, self.kv_heads, self.head_dim)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))

        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        present = (k, v)  # Retain H_kv heads; expansion is compute-only.

        repeats = self.query_heads // self.kv_heads
        k_compute = k.repeat_interleave(repeats, dim=1)
        v_compute = v.repeat_interleave(repeats, dim=1)
        scores = q @ k_compute.transpose(-2, -1) / self.head_dim**0.5

        past_len = k.size(2) - query_len
        q_positions = past_len + torch.arange(query_len, device=x.device)
        k_positions = torch.arange(k.size(2), device=x.device)
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        weights = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        output = weights @ v_compute
        output = output.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.out_proj(output), present


class ToyLatentKVAttention(nn.Module):
    """Didactic MLA-style attention that caches a latent plus shared RoPE key.

    Production MLA absorbs up-projection matrices into surrounding operations.
    This explicit version leaves the compress/reconstruct boundary inspectable.
    """

    def __init__(
        self, d_model: int, query_heads: int, latent_rank: int, rope_dim: int
    ) -> None:
        super().__init__()
        if d_model % query_heads or rope_dim % 2:
            raise ValueError("invalid head or rotary dimension")
        self.query_heads = query_heads
        self.head_dim = d_model // query_heads
        self.latent_rank = latent_rank
        self.rope_dim = rope_dim
        self.q_content = nn.Linear(d_model, d_model, bias=False)
        self.q_rope = nn.Linear(d_model, query_heads * rope_dim, bias=False)
        self.kv_down = nn.Linear(d_model, latent_rank, bias=False)
        self.k_up = nn.Linear(latent_rank, d_model, bias=False)
        self.v_up = nn.Linear(latent_rank, d_model, bias=False)
        self.k_rope = nn.Linear(d_model, rope_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        frequencies, _ = inverse_frequencies(rope_dim)
        self.register_buffer("rope_frequencies", frequencies.float(), persistent=False)

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, query_len, _ = x.shape
        latent = self.kv_down(x)
        rope_key = self.k_rope(x)
        if cache is not None:
            latent = torch.cat((cache[0], latent), dim=1)
            rope_key = torch.cat((cache[1], rope_key), dim=1)
        present = (latent, rope_key)

        key_len = latent.size(1)
        q_c = self.q_content(x).view(batch, query_len, self.query_heads, self.head_dim)
        q_r = self.q_rope(x).view(batch, query_len, self.query_heads, self.rope_dim)
        k_c = self.k_up(latent).view(batch, key_len, self.query_heads, self.head_dim)
        value = self.v_up(latent).view(batch, key_len, self.query_heads, self.head_dim)
        past_len = key_len - query_len
        q_positions = past_len + torch.arange(query_len, device=x.device)
        k_positions = torch.arange(key_len, device=x.device)
        q_r = apply_rope(
            q_r.transpose(1, 2), q_positions, self.rope_frequencies
        ).transpose(1, 2)
        k_r = apply_rope(rope_key, k_positions, self.rope_frequencies)
        k_r = k_r.unsqueeze(2).expand(-1, -1, self.query_heads, -1)
        query = torch.cat((q_c, q_r), dim=-1).transpose(1, 2)
        key = torch.cat((k_c, k_r), dim=-1).transpose(1, 2)
        value = value.transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / query.size(-1) ** 0.5
        allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        weights = F.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.out_proj(output), present

    @staticmethod
    def cache_scalars(cache: tuple[Tensor, Tensor]) -> int:
        return sum(tensor.numel() for tensor in cache)
