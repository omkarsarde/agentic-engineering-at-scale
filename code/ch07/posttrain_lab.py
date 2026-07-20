"""Deterministic CPU lab for SFT, reward modeling, REINFORCE, and DPO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
FEATURES = [
    [[1.0, 1.0, .25], [1.0, 1.0, 1.0], [0.0, 1.0, .70], [.1, .8, .15]],
    [[1.0, 1.0, .30], [1.0, 1.0, 1.0], [.1, .7, .65], [0.0, .9, .15]],
    [[.9, 0.0, .55], [1.0, 1.0, .40], [1.0, 1.0, 1.0], [.1, 1.0, .20]],
    [[1.0, 1.0, .20], [.7, 1.0, .45], [.2, 1.0, .65], [.1, .9, .15]],
]
TARGETS = [0, 0, 1, 0]
BASE_LOGITS = [
    [0.0, .6, .8, -.2], [0.0, .7, .5, -.2],
    [.8, 0.0, .4, -.2], [0.0, .6, .7, -.2],
]
SPEC_WEIGHTS = [2.4, 2.0, -.3]
ANNOTATOR_WEIGHTS = [2.4, 2.0, 1.5]
def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]

def train_sft(logits: list[list[float]], steps: int = 30, lr: float = 1.0) -> list[float]:
    """Fit one demonstrated response per prompt with cross-entropy."""
    history = []
    for step in range(steps):
        loss = 0.0
        for prompt, target in enumerate(TARGETS):
            probs = softmax(logits[prompt])
            loss -= math.log(probs[target])
            for answer in range(len(probs)):
                grad = probs[answer] - float(answer == target)
                logits[prompt][answer] -= lr * grad / len(TARGETS)
        if step in {0, 9, 19, steps - 1}:
            history.append(loss / len(TARGETS))
    return history

def collect_preferences(count: int = 512, seed: int = 7) -> list[tuple[int, int, int, bool]]:
    """Create pairwise labels with visible length and left-position biases."""
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        prompt = rng.randrange(len(FEATURES))
        left, right = rng.sample(range(len(FEATURES[prompt])), 2)
        scores = [dot(item, ANNOTATOR_WEIGHTS) for item in FEATURES[prompt]]
        p_left = 1.0 / (1.0 + math.exp(-(scores[left] - scores[right] + 1.0)))
        chose_left = rng.random() < p_left
        chosen, rejected = (left, right) if chose_left else (right, left)
        rows.append((prompt, chosen, rejected, chose_left))
    return rows

def train_reward_model(
    rows: list[tuple[int, int, int, bool]], steps: int = 250, lr: float = .2
) -> tuple[list[float], list[float]]:
    """Fit a linear Bradley-Terry reward model."""
    weights = [0.0, 0.0, 0.0]
    history = []
    for step in range(steps):
        gradient = [0.0, 0.0, 0.0]
        loss = 0.0
        for prompt, chosen, rejected, _ in rows:
            delta = [
                FEATURES[prompt][chosen][i] - FEATURES[prompt][rejected][i]
                for i in range(3)
            ]
            probability = 1.0 / (1.0 + math.exp(-dot(weights, delta)))
            loss -= math.log(max(probability, 1e-12))
            for i in range(3):
                gradient[i] += (probability - 1.0) * delta[i]
        for i in range(3):
            weights[i] -= lr * gradient[i] / len(rows)
        if step in {0, 49, 149, steps - 1}:
            history.append(loss / len(rows))
    return weights, history

def reinforce_check(lr: float = .5) -> tuple[float, float]:
    """Take one exact REINFORCE step and return rewarded-action probability."""
    logits = [0.0, 0.0, 0.0, 0.0]
    before = softmax(logits)[0]
    probs = softmax(logits)
    advantage = 1.0 - .25
    for action in range(4):
        grad_loss = -advantage * (float(action == 0) - probs[action])
        logits[action] -= lr * grad_loss
    return before, softmax(logits)[0]

def train_dpo(
    logits: list[list[float]], reference: list[list[float]],
    rows: list[tuple[int, int, int, bool]], steps: int = 300,
    lr: float = .8, beta: float = .4,
) -> list[float]:
    """Optimize completion-level DPO against a frozen SFT reference."""
    history = []
    for step in range(steps):
        gradient = [[0.0] * 4 for _ in range(4)]
        loss = 0.0
        for prompt, chosen, rejected, _ in rows:
            policy_gap = logits[prompt][chosen] - logits[prompt][rejected]
            reference_gap = reference[prompt][chosen] - reference[prompt][rejected]
            margin = beta * (policy_gap - reference_gap)
            probability = 1.0 / (1.0 + math.exp(-margin))
            loss -= math.log(max(probability, 1e-12))
            derivative = beta * (probability - 1.0)
            gradient[prompt][chosen] += derivative
            gradient[prompt][rejected] -= derivative
        for prompt in range(4):
            for answer in range(4):
                logits[prompt][answer] -= lr * gradient[prompt][answer] / len(rows)
        if step in {0, 49, 149, steps - 1}:
            history.append(loss / len(rows))
    return history

def policy_metrics(logits: list[list[float]]) -> dict[str, float]:
    probabilities = [softmax(row) for row in logits]
    top1 = sum(max(range(4), key=logits[prompt].__getitem__) == TARGETS[prompt]
               for prompt in range(4)) / 4
    utility = sum(
        sum(probabilities[p][a] * dot(FEATURES[p][a], SPEC_WEIGHTS) for a in range(4))
        for p in range(4)
    ) / 4
    length = sum(
        sum(probabilities[p][a] * FEATURES[p][a][2] for a in range(4))
        for p in range(4)
    ) / 4
    return {"spec_top1": top1, "expected_utility": utility, "normalized_length": length}

def run_experiment(seed: int = 7) -> dict[str, object]:
    """Run the complete deterministic post-training fixture."""
    base = copy.deepcopy(BASE_LOGITS)
    sft = copy.deepcopy(base)
    sft_loss = train_sft(sft)
    preferences = collect_preferences(seed=seed)
    reward_weights, reward_loss = train_reward_model(preferences)
    dpo = copy.deepcopy(sft)
    dpo_loss = train_dpo(dpo, sft, preferences)
    longer = sum(FEATURES[p][c][2] > FEATURES[p][r][2] for p, c, r, _ in preferences)
    before, after = reinforce_check()
    return {
        "seed": seed,
        "preference_rows": len(preferences),
        "left_choice_rate": sum(row[3] for row in preferences) / len(preferences),
        "longer_choice_rate": longer / len(preferences),
        "reward_weights": dict(zip(("task", "safety", "length"), reward_weights)),
        "reinforce_probability": {"before": before, "after": after},
        "loss": {"sft": sft_loss, "reward": reward_loss, "dpo": dpo_loss},
        "stages": {"base": policy_metrics(base), "sft": policy_metrics(sft),
                   "dpo": policy_metrics(dpo)},
    }

def write_svg(summary: dict[str, object], path: Path) -> None:
    """Write a dependency-free, print-safe numeric figure."""
    stages = summary["stages"]
    names = ["base", "sft", "dpo"]
    specs = [("expected_utility", "Specification utility (fixture points)", 0.0, 5.0),
             ("normalized_length", "Expected normalized length", 0.0, 1.0)]
    chunks = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">',
              '<rect width="900" height="360" fill="white"/>',
              '<style>text{font:14px sans-serif;fill:#111}.t{font:bold 16px sans-serif}.a{stroke:#111;stroke-width:1}.l{fill:none;stroke:#222;stroke-width:3}</style>']
    for panel, (key, title, low, high) in enumerate(specs):
        left = 55 + panel * 445
        chunks += [f'<text class="t" x="{left}" y="28">{title}</text>',
                   f'<line class="a" x1="{left}" y1="300" x2="{left+350}" y2="300"/>',
                   f'<line class="a" x1="{left}" y1="55" x2="{left}" y2="300"/>']
        points = []
        for i, name in enumerate(names):
            value = stages[name][key]
            x = left + 60 + i * 120
            y = 300 - (value - low) / (high - low) * 245
            points.append(f"{x},{y:.1f}")
            chunks += [f'<circle cx="{x}" cy="{y:.1f}" r="6" fill="white" stroke="#111" stroke-width="3"/>',
                       f'<text x="{x-22}" y="325">{name.upper()}</text>',
                       f'<text x="{x-20}" y="{y-12:.1f}">{value:.3f}</text>']
        chunks.append(f'<polyline class="l" points="{" ".join(points)}"/>')
        chunks += [f'<text x="{left-8}" y="304" text-anchor="end">{low:g}</text>',
                   f'<text x="{left-8}" y="60" text-anchor="end">{high:g}</text>']
    chunks.append('</svg>')
    path.write_text("\n".join(chunks), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--svg", type=Path)
    args = parser.parse_args()
    result = run_experiment(args.seed)
    if args.svg:
        write_svg(result, args.svg)
    print(json.dumps(result, indent=2, sort_keys=True))
