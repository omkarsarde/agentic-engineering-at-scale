"""Batching, held-out scoring, and paired-bootstrap helpers for the toy run."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from chapter2_adapter import BytePairTokenizer, TinyGPT


def batch(
    tokens: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a seeded next-token batch."""

    high = tokens.numel() - block_size - 1
    if high < 1:
        raise ValueError("training corpus is shorter than one context window")
    starts = torch.randint(high, (batch_size,), generator=generator)
    inputs = torch.stack([tokens[start : start + block_size] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


@torch.inference_mode()
def evaluation_loss(model: TinyGPT, tokens: torch.Tensor, windows: int = 12) -> float:
    """Average fixed-window next-token loss on disjoint text."""

    block = model.config.block_size
    starts = torch.linspace(0, max(0, tokens.numel() - block - 1), windows).long()
    losses = []
    model.eval()
    for start in starts:
        inputs = tokens[start : start + block].unsqueeze(0)
        targets = tokens[start + 1 : start + block + 1].unsqueeze(0)
        _, loss, _ = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss))
    return float(np.mean(losses))


@torch.inference_mode()
def candidate_score(
    model: TinyGPT, tokenizer: BytePairTokenizer, prompt: str, completion: str
) -> float:
    """Return length-normalized completion log probability."""

    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(completion)
    ids = (prompt_ids + completion_ids)[-model.config.block_size :]
    completion_length = min(len(completion_ids), len(ids) - 1)
    if completion_length < 1:
        raise ValueError("candidate completion must contain at least one token")
    inputs = torch.tensor(ids[:-1]).unsqueeze(0)
    logits, _, _ = model(inputs)
    log_probabilities = F.log_softmax(logits[0], dim=-1)
    targets = torch.tensor(ids[1:])
    selected = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    return float(selected[-completion_length:].mean())


def bootstrap_mean_interval(values: list[float], *, seed: int) -> tuple[float, float]:
    """Return a seeded 90% interval for a mean across paired seeds."""

    if not values:
        raise ValueError("at least one replicate is required")
    if len(values) == 1:
        return values[0], values[0]
    generator = np.random.default_rng(seed)
    array = np.asarray(values)
    means = array[generator.integers(0, len(values), size=(2_000, len(values)))].mean(1)
    low, high = np.quantile(means, (0.05, 0.95))
    return float(low), float(high)
