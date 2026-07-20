"""Deterministic behavior-cloning and group-relative RL on a tool-use gym."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
from typing import Iterable


ACTIONS = ("finish", "lookup", "request_approval", "refund")
ACTION_UNITS = {"finish": 2, "lookup": 8, "request_approval": 12, "refund": 10}


@dataclass(frozen=True)
class Task:
    task_id: str
    amount: int
    start_state: str = "start"


@dataclass(frozen=True)
class Transition:
    observation: str
    action: str
    behavior_logp: float
    behavior_version: int


@dataclass
class Trajectory:
    task_id: str
    transitions: list[Transition] = field(default_factory=list)
    success: bool = False
    policy_violation: bool = False
    duplicate_effect: bool = False
    environment_units: int = 0


@dataclass(frozen=True)
class Score:
    reward: float
    eligible: bool
    outcome: float
    policy: float
    effects: float
    cost: float


class TabularPolicy:
    """A tiny softmax policy whose states stand in for model contexts."""

    def __init__(self) -> None:
        self.logits: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.version = 0

    def probabilities(self, observation: str) -> list[float]:
        values = self.logits[observation]
        peak = max(values)
        weights = [math.exp(value - peak) for value in values]
        total = sum(weights)
        return [weight / total for weight in weights]

    def choose(self, observation: str, rng: random.Random, *, greedy: bool) -> tuple[str, float]:
        probabilities = self.probabilities(observation)
        index = max(range(len(ACTIONS)), key=probabilities.__getitem__) if greedy else rng.choices(
            range(len(ACTIONS)), weights=probabilities, k=1
        )[0]
        return ACTIONS[index], math.log(probabilities[index])

    def update(self, observation: str, action: str, weight: float) -> None:
        probabilities = self.probabilities(observation)
        selected = ACTIONS.index(action)
        for index, probability in enumerate(probabilities):
            self.logits[observation][index] += weight * ((index == selected) - probability)


class RefundGym:
    """Resettable refund environment with an intentionally permissive effect tool."""

    def reset(self, task: Task) -> str:
        self.task = task
        self.state = task.start_state
        self.done = False
        self.refunded = False
        self.violation = False
        self.effects = 0
        return self.state

    def step(self, action: str) -> str:
        if self.done:
            raise RuntimeError("step after terminal state")
        high = self.task.amount > 50
        if self.state in {"start", "recover"} and action == "lookup":
            self.state = "known_high" if high else "known_low"
        elif self.state == "known_high" and action == "request_approval":
            self.state = "approved"
        elif self.state in {"known_low", "approved"} and action == "refund":
            self.refunded, self.state, self.effects = True, "refunded", self.effects + 1
        elif self.state == "known_high" and action == "refund":
            self.refunded, self.state, self.effects = True, "refunded", self.effects + 1
            self.violation = True
        elif self.state == "refunded" and action == "refund":
            self.effects += 1
        elif self.state == "refunded" and action == "finish":
            self.done = True
        elif action == "finish":
            self.done = True
        else:
            self.state = "recover"
        return "terminal" if self.done else self.state


def expert_actions(task: Task) -> list[str]:
    middle = ["request_approval"] if task.amount > 50 else []
    return ["lookup", *middle, "refund", "finish"]


def behavior_clone(policy: TabularPolicy, tasks: Iterable[Task], epochs: int = 40) -> None:
    examples: list[tuple[str, str]] = []
    for task in tasks:
        gym = RefundGym()
        observation = gym.reset(task)
        for action in expert_actions(task):
            examples.append((observation, action))
            observation = gym.step(action)
    for _ in range(epochs):
        for observation, action in examples:
            policy.update(observation, action, 0.25)
    policy.version += 1


def rollout(policy: TabularPolicy, task: Task, rng: random.Random, *, greedy: bool = False) -> Trajectory:
    gym, trajectory = RefundGym(), Trajectory(task.task_id)
    observation = gym.reset(task)
    for _ in range(7):
        action, logp = policy.choose(observation, rng, greedy=greedy)
        trajectory.transitions.append(Transition(observation, action, logp, policy.version))
        trajectory.environment_units += ACTION_UNITS[action]
        observation = gym.step(action)
        if gym.done:
            break
    trajectory.success = gym.done and gym.refunded
    trajectory.policy_violation = gym.violation
    trajectory.duplicate_effect = gym.effects > 1
    return trajectory


def score_trajectory(trajectory: Trajectory) -> Score:
    outcome = 1.0 if trajectory.success else 0.0
    policy = -2.0 if trajectory.policy_violation else 0.0
    effects = -1.0 if trajectory.duplicate_effect else 0.0
    cost = -0.03 * len(trajectory.transitions)
    eligible = policy == 0.0 and effects == 0.0
    return Score(outcome + policy + effects + cost, eligible, outcome, policy, effects, cost)


def grouped_advantages(trajectories: list[Trajectory]) -> dict[tuple[int, int], float]:
    """Estimate local advantages from repeated visits to the same task state."""
    grouped: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    scores = [score_trajectory(trajectory).reward for trajectory in trajectories]
    for trajectory_index, trajectory in enumerate(trajectories):
        for turn, transition in enumerate(trajectory.transitions):
            grouped[(trajectory.task_id, transition.observation)].append(
                (trajectory_index, turn, scores[trajectory_index])
            )
    advantages: dict[tuple[int, int], float] = {}
    for visits in grouped.values():
        rewards = [reward for _, _, reward in visits]
        mean, spread = fmean(rewards), pstdev(rewards)
        for trajectory_index, turn, reward in visits:
            advantages[(trajectory_index, turn)] = (reward - mean) / (spread + 1e-6)
    return advantages


def train_group_relative(
    policy: TabularPolicy,
    tasks: list[Task],
    rng: random.Random,
    rounds: int = 24,
    group_size: int = 12,
) -> dict[str, object]:
    history, units = [], {"environment": 0, "policy_sampling": 0, "update": 0}
    vetoed_rollouts = vetoed_updates = max_policy_lag = 0
    for _ in range(rounds):
        batch_version = policy.version
        trajectories = [rollout(policy, task, rng) for task in tasks for _ in range(group_size)]
        max_policy_lag = max(
            max_policy_lag,
            max(batch_version - step.behavior_version for item in trajectories for step in item.transitions),
        )
        units["environment"] += sum(item.environment_units for item in trajectories)
        units["policy_sampling"] += 3 * sum(len(item.transitions) for item in trajectories)
        advantages = grouped_advantages(trajectories)
        vetoed = {index for index, item in enumerate(trajectories) if not score_trajectory(item).eligible}
        updated: set[int] = set()
        vetoed_rollouts += len(vetoed)
        for _epoch in range(2):
            for index, trajectory in enumerate(trajectories):
                if index in vetoed:
                    continue
                for turn, transition in enumerate(trajectory.transitions):
                    advantage = advantages[(index, turn)]
                    current = math.log(policy.probabilities(transition.observation)[ACTIONS.index(transition.action)])
                    ratio = math.exp(current - transition.behavior_logp)
                    if (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < 0.8):
                        continue
                    policy.update(transition.observation, transition.action, 0.04 * advantage)
                    updated.add(index)
                    units["update"] += 1
        vetoed_updates += len(vetoed & updated)
        policy.version += 1
        history.append(sum(rollout(policy, task, rng, greedy=True).success for task in tasks) / len(tasks))
    return {
        "history": history,
        "work_units": units,
        "vetoed_rollouts": vetoed_rollouts,
        "vetoed_updates": vetoed_updates,
        "max_policy_lag": max_policy_lag,
    }


def evaluate(policy: TabularPolicy, tasks: list[Task]) -> dict[str, object]:
    rng = random.Random(0)
    trajectories = [rollout(policy, task, rng, greedy=True) for task in tasks]
    return {
        "successes": sum(item.success and score_trajectory(item).eligible for item in trajectories),
        "tasks": len(tasks),
        "actions": {item.task_id: [step.action for step in item.transitions] for item in trajectories},
    }


def run_experiment(seed: int = 7, rounds: int = 24) -> dict[str, object]:
    expert_tasks = [Task("expert-low", 20), Task("expert-high", 80)]
    training_tasks = [Task("train-low", 20), Task("train-high", 80), Task("recover-low", 20, "recover"), Task("recover-high", 80, "recover")]
    evaluation_tasks = [Task("eval-low", 25), Task("eval-high", 90), Task("eval-recover-low", 25, "recover"), Task("eval-recover-high", 90, "recover")]
    policy = TabularPolicy()
    behavior_clone(policy, expert_tasks)
    before = evaluate(policy, evaluation_tasks)
    training = train_group_relative(policy, training_tasks, random.Random(seed), rounds=rounds)
    after = evaluate(policy, evaluation_tasks)
    return {"seed": seed, "rounds": rounds, "bc": before, "bc_plus_rl": after, "training": training}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    result = run_experiment(args.seed, args.rounds)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    training = result["training"]
    complete = result["bc_plus_rl"]["successes"] == result["bc_plus_rl"]["tasks"]
    safe = training["vetoed_updates"] == 0 and training["max_policy_lag"] == 0
    return 0 if complete and safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
