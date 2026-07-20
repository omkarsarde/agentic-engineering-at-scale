"""A compact decoder-only Transformer with an exact KV-cache path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LayerCache = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 48
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: float = 8 / 3


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        past_steps = 0 if past is None else past[0].size(-2)
        if past is not None:
            key = torch.cat((past[0], key), dim=-2)
            value = torch.cat((past[1], value), dim=-2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_steps + torch.arange(steps, device=x.device)[:, None]
        key_positions = torch.arange(key.size(-2), device=x.device)[None, :]
        scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
        mixed = scores.softmax(dim=-1) @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.output(mixed), (key, value)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = max(8, round(config.mlp_ratio * config.d_model))
        self.up_gate = nn.Linear(config.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        update, present = self.attention(self.attention_norm(x), past)
        x = x + update
        return x + self.mlp(self.mlp_norm(x)), present


class TinyGPT(nn.Module):
    """A pre-norm, decoder-only causal language model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Learned positions keep this chapter self-contained; Chapter 3 replaces them.
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        cache: list[LayerCache] | None = None,
    ) -> tuple[Tensor, Tensor | None, list[LayerCache]]:
        """Compute causal logits, optional next-token loss, and a present cache.

        Args:
            tokens: Integer token IDs shaped ``(batch, new_steps)``.
            targets: Optional next-token IDs with the same shape.
            cache: Optional per-layer keys and values for preceding positions.

        Returns:
            Logits, mean cross-entropy loss or ``None``, and updated layer caches.

        Raises:
            ValueError: If cached plus new tokens exceed ``block_size``.
        """

        _, steps = tokens.shape
        past_steps = 0 if cache is None else cache[0][0].size(-2)
        if past_steps + steps > self.config.block_size:
            raise ValueError("sequence exceeds configured block_size")
        positions = torch.arange(past_steps, past_steps + steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        present: list[LayerCache] = []
        for index, block in enumerate(self.blocks):
            layer_past = None if cache is None else cache[index]
            x, layer_present = block(x, layer_past)
            present.append(layer_present)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss, present

    @torch.inference_mode()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        use_cache: bool = True,
    ) -> Tensor:
        """Generate tokens with either cached or full-prefix decoding.

        Args:
            prompt: Token IDs shaped ``(batch, prompt_steps)``.
            max_new_tokens: Number of tokens to append.
            temperature: Zero for greedy decoding; positive for sampling.
            seed: Sampling seed used when temperature is positive.
            use_cache: Reuse projected keys and values when true.

        Returns:
            Prompt and generated token IDs in one tensor.
        """

        self.eval()
        if prompt.size(1) + max_new_tokens > self.config.block_size:
            raise ValueError("generation would exceed configured block_size")
        output = prompt.clone()
        cache: list[LayerCache] | None = None
        logits: Tensor | None = None
        generator = torch.Generator(device=prompt.device).manual_seed(seed)
        for _ in range(max_new_tokens):
            model_input = output if not use_cache or cache is None else output[:, -1:]
            logits, _, cache = self(model_input, cache=cache if use_cache else None)
            next_logits = logits[:, -1]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = (next_logits / temperature).softmax(dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            output = torch.cat((output, next_token), dim=1)
        return output

    def cache_bytes(self, cache: list[LayerCache]) -> int:
        """Return storage bytes occupied by all cached keys and values."""

        return sum(tensor.numel() * tensor.element_size() for pair in cache for tensor in pair)

    def parameter_count(self) -> int:
        """Return the number of unique trainable scalar parameters."""

        return sum(parameter.numel() for parameter in self.parameters())
