"""Deterministic mechanisms for Chapter 11's customization decision lab.

The fixture is a small linear classifier, not an estimate of language-model quality.
It makes frozen-base QLoRA, logit distillation, task-vector merging, and a four-set
release gate inspectable on a CPU.  Run this file to write ``metrics.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NF4 = np.array(
    [
        -1.0, -0.6961928, -0.5250731, -0.3949175,
        -0.2844414, -0.1847734, -0.0910500, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0,
    ],
    dtype=np.float64,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return stable row-wise probabilities."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def nf4_quantize(weights: np.ndarray, group_size: int = 16) -> tuple[np.ndarray, float]:
    """Quantize then dequantize flat groups with the QLoRA NF4 codebook."""
    flat = weights.ravel()
    restored = np.empty_like(flat)
    for start in range(0, flat.size, group_size):
        group = flat[start : start + group_size]
        scale = max(float(np.max(np.abs(group))), 1e-12)
        normalized = group / scale
        codes = np.argmin(np.abs(normalized[:, None] - NF4[None, :]), axis=1)
        restored[start : start + group_size] = NF4[codes] * scale
    rmse = float(np.sqrt(np.mean((flat - restored) ** 2)))
    return restored.reshape(weights.shape), rmse


@dataclass
class LoRA:
    """One frozen linear layer plus trainable low-rank matrices."""

    base: np.ndarray
    rank: int = 2
    alpha: float = 4.0
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        out_features, in_features = self.base.shape
        self.a = rng.normal(0.0, 0.08, size=(self.rank, in_features))
        self.b = np.zeros((out_features, self.rank), dtype=np.float64)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def delta(self) -> np.ndarray:
        return self.scale * self.b @ self.a

    def logits(self, x: np.ndarray) -> np.ndarray:
        return x @ (self.base + self.delta).T

    def _apply_gradient(self, grad_delta: np.ndarray, learning_rate: float) -> None:
        grad_b = self.scale * grad_delta @ self.a.T
        grad_a = self.scale * self.b.T @ grad_delta
        self.b -= learning_rate * grad_b
        self.a -= learning_rate * grad_a

    def sft_step(self, x: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        probs = softmax(self.logits(x))
        loss = -float(np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())
        grad_logits = probs
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        self._apply_gradient(grad_logits.T @ x / len(x), learning_rate)
        return loss

    def kd_step(
        self,
        x: np.ndarray,
        teacher_logits: np.ndarray,
        temperature: float,
        learning_rate: float,
    ) -> float:
        teacher = softmax(teacher_logits, temperature)
        student = softmax(self.logits(x), temperature)
        loss = float(np.sum(teacher * np.log((teacher + 1e-12) / (student + 1e-12)), axis=1).mean())
        grad_logits = (student - teacher) * temperature / len(x)
        self._apply_gradient(grad_logits.T @ x, learning_rate)
        return loss


def samples(rng: np.random.Generator, n: int, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate features and deterministic labels from a reference classifier."""
    x = rng.normal(size=(n, weights.shape[1]))
    return x, np.argmax(x @ weights.T, axis=1)


def make_fixture(seed: int = 53) -> dict[str, object]:
    """Create two related tasks and an untouched general-capability set."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.65, size=(4, 12))
    left_a, right_a = rng.normal(size=(4, 2)), rng.normal(size=(2, 12))
    left_b, right_b = rng.normal(size=(4, 2)), rng.normal(size=(2, 12))
    mask_a = np.r_[np.ones(7), np.zeros(5)]
    mask_b = np.r_[np.zeros(5), np.ones(7)]
    target_a = 0.30 * (left_a @ right_a) * mask_a
    target_b = 0.30 * (left_b @ right_b) * mask_b
    task_a = base + target_a
    task_b = base + target_b
    general_x = rng.normal(size=(480, base.shape[1]))
    general_x[:, :7] *= 0.20
    general = (general_x, np.argmax(general_x @ base.T, axis=1))
    return {
        "base": base,
        "teacher_a": task_a,
        "teacher_b": task_b,
        "sets": {
            "train": samples(rng, 320, task_a),
            "dev": samples(rng, 240, task_a),
            "task_test": samples(rng, 320, task_a),
            "general_regression": general,
            "task_b_train": samples(rng, 320, task_b),
            "task_b_test": samples(rng, 320, task_b),
            "distill": samples(rng, 400, task_a),
        },
    }


def train_sft(adapter: LoRA, data: tuple[np.ndarray, np.ndarray], epochs: int = 90) -> list[float]:
    losses = []
    for _ in range(epochs):
        losses.append(adapter.sft_step(*data, learning_rate=0.16))
    return losses


def train_kd(adapter: LoRA, x: np.ndarray, teacher_weights: np.ndarray, epochs: int = 35) -> list[float]:
    teacher_logits = x @ teacher_weights.T
    losses = []
    for _ in range(epochs):
        losses.append(adapter.kd_step(x, teacher_logits, temperature=2.0, learning_rate=0.06))
    return losses


def accuracy(weights: np.ndarray, data: tuple[np.ndarray, np.ndarray]) -> float:
    x, labels = data
    return float(np.mean(np.argmax(x @ weights.T, axis=1) == labels))


def four_set_scores(weights: np.ndarray, sets: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    return {name: accuracy(weights, sets[name]) for name in ("train", "dev", "task_test", "general_regression")}


def release_gate(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, object]:
    task_gain = candidate["task_test"] - baseline["task_test"]
    regression_drop = baseline["general_regression"] - candidate["general_regression"]
    return {
        "task_gain": float(task_gain),
        "regression_drop": float(regression_drop),
        "min_task_gain": 0.08,
        "max_regression_drop": 0.08,
        "passed": bool(task_gain >= 0.08 and regression_drop <= 0.08),
    }


def ties_merge(deltas: list[np.ndarray], weights: list[float], density: float = 0.65) -> np.ndarray:
    """Trim small updates, elect a sign, and average sign-aligned values."""
    trimmed = []
    for delta, weight in zip(deltas, weights, strict=True):
        cutoff = np.quantile(np.abs(delta), 1.0 - density)
        trimmed.append(np.where(np.abs(delta) >= cutoff, weight * delta, 0.0))
    stack = np.stack(trimmed)
    elected = np.sign(stack.sum(axis=0))
    keep = np.sign(stack) == elected
    count = keep.sum(axis=0)
    return np.divide((stack * keep).sum(axis=0), count, out=np.zeros_like(stack[0]), where=count > 0)


def merge_sweep(
    base: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    sets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float]]:
    rows = []
    for weight_a in np.linspace(0.0, 1.0, 11):
        weights = [float(weight_a), float(1.0 - weight_a)]
        for method, delta in (
            ("linear", weights[0] * delta_a + weights[1] * delta_b),
            ("ties", ties_merge([delta_a, delta_b], weights)),
        ):
            merged = base + delta
            score_a = accuracy(merged, sets["task_test"])
            score_b = accuracy(merged, sets["task_b_test"])
            rows.append(
                {
                    "method": method,
                    "weight_a": float(weight_a),
                    "task_a": score_a,
                    "task_b": score_b,
                    "task_mean": float((score_a + score_b) / 2),
                    "regression": accuracy(merged, sets["general_regression"]),
                }
            )
    return rows


def run_lab() -> dict[str, object]:
    fixture = make_fixture()
    full_base = fixture["base"]
    sets = fixture["sets"]
    quantized_base, quantization_rmse = nf4_quantize(full_base, group_size=16)

    adapter_a = LoRA(quantized_base, seed=1)
    sft_losses = train_sft(adapter_a, sets["train"])
    sft_weights = quantized_base + adapter_a.delta

    adapter_kd = LoRA(quantized_base, seed=1)
    adapter_kd.a, adapter_kd.b = adapter_a.a.copy(), adapter_a.b.copy()
    x_distill, _ = sets["distill"]
    kd_losses = train_kd(adapter_kd, x_distill, fixture["teacher_a"])
    kd_weights = quantized_base + adapter_kd.delta

    adapter_b = LoRA(quantized_base, seed=2)
    train_sft(adapter_b, sets["task_b_train"])
    sweep = merge_sweep(quantized_base, adapter_kd.delta, adapter_b.delta, sets)
    best_by_method = {
        method: max((row for row in sweep if row["method"] == method), key=lambda row: row["task_mean"])
        for method in ("linear", "ties")
    }

    baseline = four_set_scores(quantized_base, sets)
    sft = four_set_scores(sft_weights, sets)
    distilled = four_set_scores(kd_weights, sets)
    return {
        "fixture": {"kind": "synthetic linear proxy", "seed": 53, "nf4_group_size": 16},
        "quantization_rmse": quantization_rmse,
        "scores": {"base_nf4": baseline, "qlora_sft": sft, "logit_kd": distilled},
        "gate": release_gate(baseline, distilled),
        "losses": {
            "sft_first": sft_losses[0], "sft_last": sft_losses[-1],
            "kd_first": kd_losses[0], "kd_last": kd_losses[-1],
        },
        "merge_sweep": sweep,
        "merge_best": best_by_method,
    }


if __name__ == "__main__":
    metrics = run_lab()
    destination = Path(__file__).with_name("metrics.json")
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["scores"], indent=2))
    print(json.dumps(metrics["gate"], indent=2))
    print(f"wrote {destination}")
