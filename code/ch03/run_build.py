"""Run the deterministic KV-accounting and rotary-retrieval build."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
import torch

from kv_math import GroupedQueryAttention, KVConfig, ToyLatentKVAttention, kv_bytes
from rope import apply_rope, inverse_frequencies


matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
CONTEXTS = (4_096, 32_768, 131_072)


def _dtype_bytes(name: str) -> int:
    try:
        return {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}[name]
    except KeyError as error:
        raise ValueError(f"unsupported cache dtype: {name}") from error


def load_fixture(path: Path) -> tuple[KVConfig, dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    common = dict(
        name=raw["_fixture_name"],
        layers=raw["num_hidden_layers"],
        query_heads=raw["num_attention_heads"],
        bytes_per_scalar=_dtype_bytes(raw["torch_dtype"]),
    )
    if "kv_lora_rank" in raw:
        config = KVConfig(
            **common,
            head_dim=raw["v_head_dim"],
            latent_rank=raw["kv_lora_rank"],
            rope_key_dim=raw["qk_rope_head_dim"],
        )
    else:
        config = KVConfig(
            **common,
            head_dim=raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
            kv_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        )
    return config, raw


def retrieval_probe(
    method: str,
    *,
    seed: int = 1_700,
    trials: int = 256,
    context: int = 128,
    original_context: int = 32,
    dim: int = 32,
) -> list[dict[str, float | int | str]]:
    """Measure a content needle under rotary phase shifts, without an LLM."""

    factor = context / original_context
    frequencies, magnitude = inverse_frequencies(
        dim, method=method, factor=factor, original_context=original_context
    )
    rows: list[dict[str, float | int | str]] = []
    for index, depth in enumerate(torch.linspace(0, 1, 21)):
        generator = torch.Generator().manual_seed(seed + index)
        content = torch.randn(trials, dim, generator=generator)
        query = content + 0.15 * torch.randn(trials, dim, generator=generator)
        needle = content + 0.15 * torch.randn(trials, dim, generator=generator)
        keys = torch.randn(trials, context, dim, generator=generator)
        needle_position = round(float(depth) * (context - 2))
        keys[:, needle_position] = needle

        query_rotated = apply_rope(
            query[:, None], torch.tensor([context - 1]), frequencies, magnitude
        )[:, 0]
        keys_rotated = apply_rope(
            keys, torch.arange(context), frequencies, magnitude
        )
        scores = torch.einsum("btd,bd->bt", keys_rotated, query_rotated) / dim**0.5
        successes = int((scores.argmax(dim=1) == needle_position).sum())
        probability = successes / trials
        standard_error = math.sqrt(probability * (1 - probability) / trials)
        rows.append(
            {
                "method": method,
                "depth_percent": round(float(depth) * 100),
                "successes": successes,
                "trials": trials,
                "accuracy": probability,
                "standard_error": standard_error,
            }
        )
    return rows


def _save_figures(
    footprint_rows: list[dict[str, object]],
    retrieval_rows: list[dict[str, float | int | str]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-03"
    styles = (
        ("#2a7f9e", "o", "-"),
        ("#a14f3b", "s", "--"),
        ("#7356a8", "^", ":"),
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.1))
    for (color, marker, line_style), name in zip(
        styles, dict.fromkeys(row["name"] for row in footprint_rows), strict=True
    ):
        selected = [row for row in footprint_rows if row["name"] == name]
        x = [row["tokens"] / 1_024 for row in selected]
        y = [row["gib"] for row in selected]
        axis.plot(
            x,
            y,
            marker=marker,
            linestyle=line_style,
            color=color,
        )
        axis.annotate(
            name,
            xy=(x[-1], y[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axis.set(xlabel="Context represented in cache (Ki tokens)", ylabel="KV payload (GiB)")
    axis.set_xlim(0, 190)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "kv-footprints.svg", format="svg", metadata={"Date": None})
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.1))
    labels = {"rope": "RoPE, unscaled", "ntk": "NTK-aware", "yarn": "YaRN"}
    label_y = {"rope": 0.79, "ntk": 0.89, "yarn": 0.99}
    for (color, marker, line_style), method in zip(styles, labels, strict=True):
        selected = [row for row in retrieval_rows if row["method"] == method]
        x = [row["depth_percent"] for row in selected]
        y = [row["accuracy"] for row in selected]
        error = [1.96 * row["standard_error"] for row in selected]
        axis.plot(
            x,
            y,
            marker=marker,
            markevery=2,
            markersize=4,
            linestyle=line_style,
            color=color,
        )
        axis.fill_between(x, [max(0, a - e) for a, e in zip(y, error)], [min(1, a + e) for a, e in zip(y, error)], color=color, alpha=0.12)
        axis.annotate(
            labels[method],
            xy=(x[-2], y[-2]),
            xytext=(102, label_y[method]),
            textcoords="data",
            va="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.7},
        )
    axis.set(
        xlabel="Needle depth from start of context (%)",
        ylabel="Top-1 retrieval rate",
        xlim=(0, 125),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "rope-retrieval.svg", format="svg", metadata={"Date": None})
    plt.close(fig)


def run_build(out_dir: Path) -> dict[str, object]:
    """Generate all Chapter 3 measurements from local deterministic fixtures."""

    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_fixture(path) for path in sorted((HERE / "fixtures").glob("*/config.json"))]
    footprint_rows: list[dict[str, object]] = []
    for config, raw in loaded:
        for tokens in CONTEXTS:
            byte_count = kv_bytes(config, tokens)
            footprint_rows.append(
                {
                    "name": config.name,
                    "tokens": tokens,
                    "bytes": byte_count,
                    "gib": byte_count / 2**30,
                    "source": raw["_source"],
                    "verified_on": raw["_verified_on"],
                }
            )

    retrieval_rows = [
        row for method in ("rope", "ntk", "yarn") for row in retrieval_probe(method)
    ]
    torch.manual_seed(7)
    sample = torch.randn(1, 8, 48)
    grouped = GroupedQueryAttention(48, query_heads=4, kv_heads=2)
    grouped_output, grouped_cache = grouped(sample)
    latent = ToyLatentKVAttention(48, query_heads=4, latent_rank=8, rope_dim=4)
    latent_output, latent_cache = latent(sample)

    metrics: dict[str, object] = {
        "contexts": CONTEXTS,
        "footprints": footprint_rows,
        "retrieval": retrieval_rows,
        "mean_retrieval_accuracy": {
            method: sum(row["accuracy"] for row in retrieval_rows if row["method"] == method) / 21
            for method in ("rope", "ntk", "yarn")
        },
        "attention_checks": {
            "grouped_output_shape": list(grouped_output.shape),
            "grouped_cached_scalars": sum(tensor.numel() for tensor in grouped_cache),
            "latent_output_shape": list(latent_output.shape),
            "latent_cached_scalars": latent.cache_scalars(latent_cache),
        },
        "probe_contract": {
            "seed_family": 1_700,
            "trials_per_depth": 256,
            "context": 128,
            "original_context": 32,
            "rotary_dimension": 32,
            "claim": "synthetic attention diagnostic, not a language-model benchmark",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for filename, rows in (("kv-footprints.csv", footprint_rows), ("rope-retrieval.csv", retrieval_rows)):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _save_figures(footprint_rows, retrieval_rows, out_dir)
    return metrics


if __name__ == "__main__":
    run_build(HERE / "generated")
