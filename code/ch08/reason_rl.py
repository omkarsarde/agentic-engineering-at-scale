"""Engine-free CPU fixture for test-time search, GRPO, RLVR, and Goodhart."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random


INITIAL_LOGITS = [
    [.6, -.1, .2, -.4, -1.2],
    [0.0, -.2, .5, 0.0, -1.0],
    [-.6, -.4, .7, .2, -.8],
]
TOKEN_LENGTHS = [4, 12, 7, 5, 30]
TASKS = [(i % 3, 11 + 7 * i) for i in range(60)]


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    values = [math.exp(item - peak) for item in logits]
    return [item / sum(values) for item in values]


def sample_index(probabilities: list[float], rng: random.Random) -> int:
    threshold, cumulative = rng.random(), 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


def is_correct(action: int) -> bool:
    return action < 2


def candidate_answer(gold: int, action: int, nonce: int) -> int:
    """Map a reasoning strategy to a deterministic arithmetic answer."""
    if is_correct(action):
        return gold
    if action == 2:
        return gold - 1
    if action == 3:
        return gold + 1 + nonce % 7
    return gold + 99


def ttc_curve(seed: int = 11, trials: int = 1200) -> list[dict[str, float]]:
    """Measure coverage, plurality, and noisy-verifier selection versus tokens."""
    rows = []
    for samples in (1, 2, 4, 8, 16, 32):
        rng = random.Random(seed + samples)
        coverage = plurality = noisy_bon = token_total = 0
        for trial in range(trials):
            difficulty, gold = TASKS[trial % len(TASKS)]
            probabilities = softmax(INITIAL_LOGITS[difficulty])
            actions = [sample_index(probabilities, rng) for _ in range(samples)]
            answers = [candidate_answer(gold, action, trial + i) for i, action in enumerate(actions)]
            correct = [is_correct(action) for action in actions]
            coverage += any(correct)
            winner = max(dict.fromkeys(answers), key=answers.count)
            plurality += winner == gold
            marked = [i for i, hit in enumerate(correct) if hit != (rng.random() < .18)]
            selected = rng.choice(marked) if marked else 0
            noisy_bon += correct[selected]
            token_total += sum(TOKEN_LENGTHS[action] for action in actions)
        rows.append({
            "samples": samples,
            "mean_tokens": token_total / trials,
            "coverage": coverage / trials,
            "plurality": plurality / trials,
            "noisy_bon": noisy_bon / trials,
        })
    return rows


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Return group-relative, population-standardized advantages."""
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance == 0.0:
        return [0.0] * len(rewards)
    scale = math.sqrt(variance)
    return [(reward - mean) / scale for reward in rewards]


def reward_values(mode: str) -> list[float]:
    if mode == "exact":
        return [1.0, 1.0, 0.0, 0.0, 0.0]
    if mode == "proxy":
        return [
            float(is_correct(action)) + 2.2 * TOKEN_LENGTHS[action] / 30
            + .1 * float(action != 3)
            for action in range(5)
        ]
    raise ValueError(f"unknown reward mode: {mode}")


def snapshot(step: int, logits: list[list[float]], mode: str) -> dict[str, float]:
    distributions = [softmax(row) for row in logits]
    rewards = reward_values(mode)
    max_reward = max(rewards)
    return {
        "step": step,
        "true_accuracy": sum(p[0] + p[1] for p in distributions) / 3,
        "normalized_entropy": sum(
            -sum(value * math.log(max(value, 1e-12)) for value in p) / math.log(5)
            for p in distributions
        ) / 3,
        "mean_tokens": sum(sum(p[a] * TOKEN_LENGTHS[a] for a in range(5))
                           for p in distributions) / 3,
        "normalized_objective": sum(sum(p[a] * rewards[a] / max_reward for a in range(5))
                                    for p in distributions) / 3,
    }


def train_grpo(
    mode: str, seed: int = 23, steps: int = 120, group_size: int = 8,
    lr: float = .35, clip: float = .2, kl_beta: float = .015,
) -> list[dict[str, float]]:
    """Train one discrete policy with clipped GRPO and a reference KL."""
    rng = random.Random(seed)
    logits = copy.deepcopy(INITIAL_LOGITS)
    reference = [softmax(row) for row in logits]
    history = [snapshot(0, logits, mode)]
    checkpoints = {10, 30, 60, 90, steps}
    rewards_by_action = reward_values(mode)
    for step in range(1, steps + 1):
        difficulty = (step - 1) % 3
        old = softmax(logits[difficulty])
        actions = [sample_index(old, rng) for _ in range(group_size)]
        advantages = grpo_advantages([rewards_by_action[action] for action in actions])
        for _ in range(3):
            probabilities = softmax(logits[difficulty])
            gradient = [0.0] * 5
            for action, advantage in zip(actions, advantages):
                ratio = probabilities[action] / old[action]
                clipped = (advantage >= 0 and ratio > 1 + clip) or (
                    advantage < 0 and ratio < 1 - clip
                )
                if not clipped:
                    for index in range(5):
                        score = float(index == action) - probabilities[index]
                        gradient[index] -= advantage * ratio * score / group_size
            kl = sum(p * math.log(p / q) for p, q in zip(probabilities, reference[difficulty]))
            for index, probability in enumerate(probabilities):
                gradient[index] += kl_beta * probability * (
                    math.log(probability / reference[difficulty][index]) - kl
                )
                logits[difficulty][index] -= lr * gradient[index]
        if step in checkpoints:
            history.append(snapshot(step, logits, mode))
    return history


def run_experiment() -> dict[str, object]:
    return {
        "ttc": ttc_curve(),
        "training": {"exact": train_grpo("exact"), "proxy": train_grpo("proxy")},
        "reward_values": {"exact": reward_values("exact"), "proxy": reward_values("proxy")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = run_experiment()
    if args.compact:
        result["training"] = {key: value[-1] for key, value in result["training"].items()}
    print(json.dumps(result, indent=2, sort_keys=True))
