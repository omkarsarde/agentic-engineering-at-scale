"""Train the Chapter 2 tiny GPT and generate all measured artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
import torch

from bpe import BytePairTokenizer
from tinygpt import GPTConfig, TinyGPT


matplotlib.use("Agg")
FIXTURE_TEXT = (
    "A model reads tokens from left to right. Attention selects useful history. "
    "A residual stream carries each update. Tests guard every causal boundary. "
    "Small measurements reveal large design mistakes. Cache keys and values once. "
) * 24


def _batch(data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw contiguous next-token examples from one token stream."""

    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,), generator=generator)
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


def _schedule(step: int, steps: int, peak: float) -> float:
    """Linear warmup followed by cosine decay to ten percent of peak."""

    warmup = max(1, steps // 10)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _save_plots(losses: list[float], uniform_loss: float, cache_tokens: list[int], cache_bytes: list[int], out_dir: Path) -> None:
    """Write deterministic loss and cache-growth figures."""

    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-02"
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.plot(range(len(losses)), losses, color="#2a7f9e", label="training batch loss")
    axis.axhline(uniform_loss, color="#a14f3b", linestyle="--", label="ln(V) at uniform prediction")
    axis.set(xlabel="Optimizer step", ylabel="Cross-entropy loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss-curve.svg", format="svg", metadata={"Date": None})
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.plot(cache_tokens, cache_bytes, marker="o", color="#7356a8")
    axis.set(xlabel="Tokens represented in cache", ylabel="KV-cache storage (bytes)")
    fig.tight_layout()
    fig.savefig(out_dir / "kv-cache-growth.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def run_build(out_dir: Path, steps: int = 40, full: bool = False, text_path: Path | None = None) -> dict[str, object]:
    """Train one model and emit tokenizer, loss, sample, and cache evidence.

    Args:
        out_dir: Directory receiving JSON, text, and SVG artifacts.
        steps: Number of optimizer updates.
        full: Use a larger model and CUDA when available.
        text_path: Optional UTF-8 corpus; otherwise use the offline fixture.

        Returns:
            Metrics including loss, parameter count, and cache equivalence error.

        Raises:
            ValueError: If no optimizer steps are requested or the corpus is too short.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = text_path.read_text(encoding="utf-8") if text_path else FIXTURE_TEXT
    tokenizer = BytePairTokenizer.train(text, vocab_size=288 if full else 280)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    shape = {"block_size": 96, "d_model": 128, "n_heads": 8, "n_layers": 4} if full else {
        "block_size": 32, "d_model": 48, "n_heads": 4, "n_layers": 2
    }
    config = GPTConfig(vocab_size=tokenizer.vocab_size, **shape)
    if encoded.numel() <= config.block_size + 1:
        raise ValueError("corpus is too short for one training window")
    device = torch.device("cuda" if full and torch.cuda.is_available() else "cpu")
    model = TinyGPT(config).to(device)
    generator = torch.Generator().manual_seed(11)
    eval_x, eval_y = _batch(encoded, config.block_size, 8, generator)
    with torch.no_grad():
        _, initial, _ = model(eval_x.to(device), eval_y.to(device))
    assert initial is not None
    uniform_loss = math.log(config.vocab_size)
    if abs(initial.item() - uniform_loss) > 0.35:
        raise AssertionError("initial loss is inconsistent with near-uniform logits")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses: list[float] = []
    batch_size = 16 if full else 8
    for step in range(steps):
        inputs, targets = _batch(encoded, config.block_size, batch_size, generator)
        for group in optimizer.param_groups:
            group["lr"] = _schedule(step, steps, 3e-3)
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(inputs.to(device), targets.to(device))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    prompt_ids = tokenizer.encode("A model reads")
    prompt = torch.tensor([prompt_ids], device=device)
    sample_lines: list[str] = []
    for temperature in (0.0, 0.7, 1.1):
        generated = model.generate(prompt, min(24, config.block_size - prompt.size(1)), temperature, seed=19)
        sample_lines.append(f"temperature={temperature}\n{tokenizer.decode(generated[0].tolist(), errors='replace')}\n")

    context = prompt
    with torch.inference_mode():
        full_logits, _, cache = model(context)
        cache_tokens = [context.size(1)]
        cache_bytes = [model.cache_bytes(cache)]
        errors: list[float] = [0.0]
        for _ in range(min(12, config.block_size - context.size(1))):
            next_token = full_logits[:, -1].argmax(dim=-1, keepdim=True)
            context = torch.cat((context, next_token), dim=1)
            cached_logits, _, cache = model(next_token, cache=cache)
            full_logits, _, _ = model(context)
            errors.append((cached_logits[:, -1] - full_logits[:, -1]).abs().max().item())
            cache_tokens.append(context.size(1))
            cache_bytes.append(model.cache_bytes(cache))

    metrics: dict[str, object] = {
        "config": asdict(config),
        "parameters": model.parameter_count(),
        "initial_loss": initial.item(),
        "uniform_loss_ln_vocab": uniform_loss,
        "final_loss_mean_10": sum(losses[-10:]) / min(10, len(losses)),
        "max_cached_logit_error": max(errors),
        "cache_tokens": cache_tokens,
        "cache_bytes": cache_bytes,
        "device": str(device),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "tokenizer.json").write_text(json.dumps(tokenizer.as_dict(), indent=2), encoding="utf-8")
    (out_dir / "samples.txt").write_text("\n".join(sample_lines), encoding="utf-8")
    _save_plots(losses, uniform_loss, cache_tokens, cache_bytes, out_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--text", type=Path)
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (800 if args.full else 40)
    print(json.dumps(run_build(args.out_dir, steps, args.full, args.text), indent=2))


if __name__ == "__main__":
    main()
