"""Executable contracts for Chapter 3 attention, KV math, and RoPE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch03"))

from kv_math import GroupedQueryAttention, KVConfig, ToyLatentKVAttention, kv_bytes  # noqa: E402
from rope import apply_rope, inverse_frequencies, ntk_base  # noqa: E402
from run_build import load_fixture, run_build  # noqa: E402


def test_canonical_mha_and_gqa_cache_arithmetic() -> None:
    mha = KVConfig("MHA", layers=32, query_heads=32, kv_heads=32, head_dim=128, bytes_per_scalar=2)
    gqa = KVConfig("GQA", layers=32, query_heads=32, kv_heads=8, head_dim=128, bytes_per_scalar=2)
    assert kv_bytes(mha, 1) == 512 * 1_024
    assert kv_bytes(mha, 4_096) == 2 * 2**30
    assert kv_bytes(gqa, 131_072) == 16 * 2**30
    assert kv_bytes(gqa, 1) * 4 == kv_bytes(mha, 1)


def test_mla_cache_width_is_latent_plus_decoupled_rope_key() -> None:
    config = KVConfig(
        "MLA", layers=27, query_heads=16, head_dim=128, bytes_per_scalar=2,
        latent_rank=512, rope_key_dim=64,
    )
    assert config.cached_scalars_per_layer_token() == 576
    assert kv_bytes(config, 1) == 27 * 576 * 2


def test_grouped_query_dial_changes_state_not_output_shape() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 5, 32)
    caches = []
    for kv_heads in (4, 2, 1):
        layer = GroupedQueryAttention(32, query_heads=4, kv_heads=kv_heads)
        output, cache = layer(x)
        assert output.shape == x.shape
        assert cache[0].shape == (2, kv_heads, 5, 8)
        caches.append(sum(tensor.numel() for tensor in cache))
    assert caches == [640, 320, 160]


def test_grouped_query_cached_chunks_match_full_sequence() -> None:
    torch.manual_seed(5)
    layer = GroupedQueryAttention(32, query_heads=4, kv_heads=2).eval()
    x = torch.randn(1, 6, 32)
    full, _ = layer(x)
    cache = None
    pieces = []
    for index in range(x.size(1)):
        output, cache = layer(x[:, index : index + 1], cache)
        pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)


def test_toy_mla_caches_only_latent_and_shared_rotary_key() -> None:
    torch.manual_seed(11)
    layer = ToyLatentKVAttention(48, query_heads=4, latent_rank=8, rope_dim=4)
    x = torch.randn(2, 7, 48)
    output, cache = layer(x)
    assert output.shape == x.shape
    assert layer.cache_scalars(cache) == 2 * 7 * (8 + 4)


def test_rope_preserves_norm_and_depends_only_on_relative_offset() -> None:
    torch.manual_seed(13)
    q = torch.randn(1, 1, 16)
    k = torch.randn(1, 1, 16)
    frequencies, magnitude = inverse_frequencies(16)
    q_a = apply_rope(q, torch.tensor([7]), frequencies, magnitude)
    k_a = apply_rope(k, torch.tensor([19]), frequencies, magnitude)
    q_b = apply_rope(q, torch.tensor([107]), frequencies, magnitude)
    k_b = apply_rope(k, torch.tensor([119]), frequencies, magnitude)
    torch.testing.assert_close(q_a.norm(), q.norm(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close((q_a * k_a).sum(), (q_b * k_b).sum(), rtol=1e-5, atol=1e-5)


def test_scaling_changes_frequency_bands_as_designed() -> None:
    rope, _ = inverse_frequencies(16)
    pi, _ = inverse_frequencies(16, method="pi", factor=4)
    ntk, _ = inverse_frequencies(16, method="ntk", factor=4)
    yarn, magnitude = inverse_frequencies(16, method="yarn", factor=4, original_context=32)
    torch.testing.assert_close(pi, rope / 4)
    assert ntk_base(10_000, 4, 16) > 10_000
    assert ntk[0] == rope[0] and ntk[-1] < rope[-1]
    assert torch.all(yarn <= rope) and torch.all(yarn >= pi)
    assert magnitude > 1


def test_fixture_loading_and_build_outputs(tmp_path: Path) -> None:
    fixtures = sorted((ROOT / "code" / "ch03" / "fixtures").glob("*/config.json"))
    assert len(fixtures) == 3
    assert all(load_fixture(path)[1]["_verified_on"] == "2026-07-19" for path in fixtures)
    metrics = run_build(tmp_path)
    means = metrics["mean_retrieval_accuracy"]
    assert means["yarn"] > means["rope"]
    assert metrics["attention_checks"]["latent_cached_scalars"] == 96
    for name in ("metrics.json", "kv-footprints.csv", "rope-retrieval.csv", "kv-footprints.svg", "rope-retrieval.svg"):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["probe_contract"]["claim"].startswith("synthetic")
