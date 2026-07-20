"""Executable invariants for the Chapter 2 tokenizer and tiny GPT."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch02"))

from bpe import BytePairTokenizer  # noqa: E402
from tinygpt import GPTConfig, TinyGPT  # noqa: E402
from train import run_build  # noqa: E402


def fixture_model() -> TinyGPT:
    torch.manual_seed(3)
    return TinyGPT(GPTConfig(vocab_size=280, block_size=20, d_model=32, n_heads=4, n_layers=2)).eval()


def test_bpe_round_trips_and_has_no_unknown_token_path() -> None:
    tokenizer = BytePairTokenizer.train("repeat repeat smaller pieces", 272)
    for text in ("", "repeat", "unseen 🧪 snow 雪", "line one\nline two"):
        ids = tokenizer.encode(text)
        assert all(0 <= token < tokenizer.vocab_size for token in ids)
        assert tokenizer.decode(ids) == text
    assert all(token < 256 for token in tokenizer.encode("🧪"))


def test_bpe_merge_order_is_deterministic() -> None:
    first = BytePairTokenizer.train("banana bandana banana", 270)
    second = BytePairTokenizer.train("banana bandana banana", 270)
    assert first.merges == second.merges


def test_causal_mask_blocks_future_information_exactly() -> None:
    model = fixture_model()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 90, 91]])
    first_logits, _, _ = model(first)
    changed_logits, _, _ = model(changed_future)
    torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3], rtol=0, atol=0)


def test_random_initial_loss_is_near_log_vocab_size() -> None:
    model = fixture_model()
    tokens = torch.arange(16).view(2, 8) % model.config.vocab_size
    targets = (tokens + 1) % model.config.vocab_size
    _, loss, _ = model(tokens, targets)
    assert loss is not None
    assert abs(loss.item() - math.log(model.config.vocab_size)) < 0.35


def test_cached_and_uncached_logits_match() -> None:
    model = fixture_model()
    context = torch.tensor([[1, 2, 3, 4]])
    _, _, cache = model(context)
    for token in (5, 6, 7):
        next_token = torch.tensor([[token]])
        context = torch.cat((context, next_token), dim=1)
        cached, _, cache = model(next_token, cache=cache)
        uncached, _, _ = model(context)
        torch.testing.assert_close(cached[:, -1], uncached[:, -1], rtol=1e-5, atol=1e-6)
    assert torch.equal(model.generate(context, 4, use_cache=True), model.generate(context, 4, use_cache=False))


def test_cache_storage_grows_by_one_fixed_slice_per_token() -> None:
    model = fixture_model()
    _, _, cache = model(torch.tensor([[1]]))
    sizes = [model.cache_bytes(cache)]
    for token in (2, 3, 4):
        _, _, cache = model(torch.tensor([[token]]), cache=cache)
        sizes.append(model.cache_bytes(cache))
    expected_step = 2 * model.config.n_layers * model.config.d_model * 4
    assert [right - left for left, right in zip(sizes, sizes[1:])] == [expected_step] * 3


def test_weight_tying_and_build_outputs(tmp_path: Path) -> None:
    model = fixture_model()
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    metrics = run_build(tmp_path, steps=4)
    assert abs(metrics["initial_loss"] - metrics["uniform_loss_ln_vocab"]) < 0.35
    assert metrics["max_cached_logit_error"] < 1e-5
    for name in ("metrics.json", "tokenizer.json", "samples.txt", "loss-curve.svg", "kv-cache-growth.svg"):
        assert (tmp_path / name).exists()
