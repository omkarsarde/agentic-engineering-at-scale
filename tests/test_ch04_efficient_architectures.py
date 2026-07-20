"""Executable contracts for Chapter 4 sparse and efficient architectures."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch04"))

from config_reader import estimate_config  # noqa: E402
from moe_min import GatedLinearAttention, TopKRouter, mqar_capacity_curve, train_router  # noqa: E402

_BUILD_SPEC = importlib.util.spec_from_file_location(
    "ch04_run_build", ROOT / "code" / "ch04" / "run_build.py"
)
assert _BUILD_SPEC is not None and _BUILD_SPEC.loader is not None
_BUILD_MODULE = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(_BUILD_MODULE)
run_build = _BUILD_MODULE.run_build


def test_router_shapes_scores_and_top_k_load() -> None:
    torch.manual_seed(2)
    tokens = torch.randn(3, 5, 8)
    for scoring in ("softmax", "sigmoid"):
        router = TopKRouter(8, experts=4, top_k=2, scoring=scoring)
        state = router(tokens)
        assert state.probabilities.shape == (3, 5, 4)
        assert state.indices.shape == state.weights.shape == (3, 5, 2)
        torch.testing.assert_close(state.weights.sum(-1), torch.ones(3, 5))
        torch.testing.assert_close(state.load.sum(), torch.tensor(1.0))
        assert torch.isfinite(state.balance_loss) and torch.isfinite(state.z_loss)


def test_aux_free_controller_pushes_down_an_overloaded_expert() -> None:
    router = TopKRouter(4, experts=4, top_k=1)
    router.update_selection_bias(torch.tensor([0.7, 0.1, 0.1, 0.1]), rate=0.02)
    assert router.selection_bias[0] < 0
    assert torch.all(router.selection_bias[1:] > 0)
    torch.testing.assert_close(router.selection_bias.mean(), torch.tensor(0.0))


def test_balancing_prevents_seeded_router_collapse() -> None:
    unbalanced = train_router(0.0)
    balanced = train_router(0.5)
    assert max(unbalanced["load"]) > 0.95
    assert max(balanced["load"]) < 0.35
    assert min(unbalanced["accuracy"], balanced["accuracy"]) > 0.9


def test_gated_linear_attention_is_chunk_equivalent_with_fixed_state() -> None:
    torch.manual_seed(7)
    layer = GatedLinearAttention(width=12, state_width=5).eval()
    tokens = torch.randn(2, 9, 12)
    full, full_state = layer(tokens)
    recurrent = None
    pieces = []
    for index in range(tokens.size(1)):
        output, recurrent = layer(tokens[:, index : index + 1], recurrent)
        pieces.append(output)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
    assert recurrent is not None
    assert sum(tensor.numel() for tensor in recurrent) == 2 * (5 * 12 + 5)
    assert [tensor.shape for tensor in recurrent] == [tensor.shape for tensor in full_state]


def test_bounded_state_mqar_exposes_capacity_tradeoff() -> None:
    rows = mqar_capacity_curve()
    lookup = {(row["memory"], row["pairs"]): row["accuracy"] for row in rows}
    assert all(lookup[("full attention", pairs)] == 1 for pairs in (2, 8, 32, 128))
    assert lookup[("fixed state (8 slots)", 128)] < lookup[("fixed state (8 slots)", 8)]
    assert lookup[("fixed state (32 slots)", 64)] > lookup[("fixed state (8 slots)", 64)]
    assert lookup[("fixed state (32 slots)", 128)] < 0.5


def test_config_parser_reconstructs_total_counts_and_context_state() -> None:
    paths = sorted((ROOT / "code" / "ch04" / "fixtures").glob("*/config.json"))
    estimates = {estimate.name: estimate for estimate in map(estimate_config, paths)}
    assert set(estimates) == {"DeepSeek-V3", "Llama 3.1 8B", "Qwen3-Next 80B-A3B"}
    assert all(abs(estimate.total_error_percent) < 2 for estimate in estimates.values())
    assert estimates["Llama 3.1 8B"].kv_bytes_per_token == 131_072
    assert estimates["Llama 3.1 8B"].context_state_gib_32k == 4.0
    assert estimates["DeepSeek-V3"].kv_bytes_per_token == 61 * 576 * 2
    assert estimates["Qwen3-Next 80B-A3B"].kv_bytes_per_token == 12 * 2 * 2 * 256 * 2
    assert estimates["Qwen3-Next 80B-A3B"].fixed_state_bytes > 0
    assert estimates["Qwen3-Next 80B-A3B"].active_params < estimates["Qwen3-Next 80B-A3B"].total_params


def test_integrated_build_emits_machine_readable_evidence(tmp_path: Path) -> None:
    metrics = run_build(tmp_path)
    assert len(metrics["config_estimates"]) == 3
    assert len(metrics["landscape_models"]) == 8
    assert metrics["experiment_contracts"]["mqar"]["claim"].startswith("synthetic")
    for filename in (
        "metrics.json",
        "config-estimates.csv",
        "router-loads.csv",
        "mqar-capacity.csv",
        "landscape-models.csv",
        "router-load.svg",
        "fixed-state-mqar.svg",
        "landscape-total-active.svg",
    ):
        assert (tmp_path / filename).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["experiment_contracts"]["router"]["seed"] == 5
