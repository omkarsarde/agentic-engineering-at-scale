"""Evaluate action encoding, learned dynamics, and a gated embodied episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


ACTION_DIM = 7


def tokenize_action(action: np.ndarray, bins: int = 256) -> np.ndarray:
    """Quantize a normalized 7-DoF action into integer tokens."""
    action = np.asarray(action, dtype=float)
    if action.shape != (ACTION_DIM,) or bins < 2 or np.any(np.abs(action) > 1):
        raise ValueError("expected seven values in [-1, 1] and at least two bins")
    return np.rint((action + 1.0) * (bins - 1) / 2.0).astype(int)


def detokenize_action(tokens: np.ndarray, bins: int = 256) -> np.ndarray:
    """Map seven action tokens back to bin centers in [-1, 1]."""
    tokens = np.asarray(tokens, dtype=int)
    if tokens.shape != (ACTION_DIM,) or bins < 2 or np.any((tokens < 0) | (tokens >= bins)):
        raise ValueError("tokens must be seven integers inside the vocabulary")
    return 2.0 * tokens / (bins - 1) - 1.0


def roundtrip_error(action: np.ndarray, bins: int = 256) -> float:
    return float(np.max(np.abs(action - detokenize_action(tokenize_action(action, bins), bins))))


def endpoint_success(
    horizon: int, model_error: float, replan: bool, *, trials: int = 512,
    tolerance: float = 0.10, seed: int = 7,
) -> float:
    """Measure goal success under an unmodeled, episode-constant dynamics bias."""
    if horizon < 1 or model_error < 0:
        raise ValueError("horizon must be positive and model_error nonnegative")
    biases = np.random.default_rng(seed).normal(0.0, model_error, size=(trials, 2))
    goal = np.array([1.0, 0.5])
    successes = 0
    for bias in biases:
        state = np.zeros(2)
        fixed_action = goal / horizon
        for step in range(horizon):
            remaining = horizon - step
            action = (goal - state) / remaining if replan else fixed_action
            state = state + action + bias
        successes += np.linalg.norm(state - goal) <= tolerance
    return round(successes / trials, 4)


def horizon_sweep(
    horizons: tuple[int, ...] = (2, 4, 8, 12, 16, 24), model_error: float = 0.012,
) -> dict[str, list[float] | list[int]]:
    return {
        "horizon_steps": list(horizons),
        "open_loop_success": [endpoint_success(h, model_error, False) for h in horizons],
        "receding_horizon_success": [endpoint_success(h, model_error, True) for h in horizons],
    }


def plot_horizon(path: Path, model_error: float = 0.012) -> None:
    """Render the synthetic decision-horizon experiment as an SVG."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-31"
    data = horizon_sweep(model_error=model_error)
    x = data["horizon_steps"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(x, data["open_loop_success"], "o--", label="Plan once (open loop)")
    ax.plot(x, data["receding_horizon_success"], "s-", label="Observe and replan")
    ax.set(xlabel="Planning horizon (control steps)", ylabel="Final-state success rate",
           ylim=(-0.03, 1.03), xlim=(min(x), max(x) + 7))
    ax.grid(alpha=0.25)
    for label, values in (("plan once", data["open_loop_success"]),
                          ("observe + replan", data["receding_horizon_success"])):
        ax.text(max(x) + 0.7, values[-1], f"{label}: {values[-1]:.0%}", va="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


class ReversibilityGate:
    """Require explicit approval outside a deterministic action envelope."""

    def __init__(self, max_displacement: float = 0.18, max_force_n: float = 8.0):
        self.max_displacement, self.max_force_n, self.hits = max_displacement, max_force_n, 0

    def authorize(self, action: np.ndarray, approve: Callable[[dict[str, Any]], bool] | None) -> bool:
        displacement = float(np.linalg.norm(action[:3]))
        predicted_force = float(max(0.0, -action[6]) * 12.0)
        reasons = tuple(name for name, failed in (
            ("displacement", displacement > self.max_displacement),
            ("predicted contact force", predicted_force > self.max_force_n),
        ) if failed)
        if not reasons:
            return True
        self.hits += 1
        proposal = {"action": tuple(map(float, action)), "displacement": displacement,
                    "predicted_force_n": predicted_force, "reasons": reasons}
        return bool(approve and approve(proposal))


class MiniTabletopEnv:
    """Deterministic CPU fallback; it validates the harness, not VLA quality."""

    instruction = "Move the red object into the target tray."

    def reset(self, seed: int = 0) -> dict[str, Any]:
        jitter = np.random.default_rng(seed).uniform(-0.025, 0.025, size=2)
        self.gripper = np.array([0.0, -0.45])
        self.object = np.array([-0.35, -0.05]) + jitter
        self.goal, self.held, self.steps = np.array([0.40, 0.25]), False, 0
        return self.observe()

    def observe(self) -> dict[str, Any]:
        return {"gripper": self.gripper.copy(), "object": self.object.copy(),
                "goal": self.goal.copy(), "held": self.held}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], bool, bool]:
        action = np.asarray(action, dtype=float)
        self.gripper += np.clip(action[:2], -0.12, 0.12)
        if action[6] < -0.5 and np.linalg.norm(self.gripper - self.object) < 0.09:
            self.held = True
        if self.held:
            self.object = self.gripper.copy()
        if action[6] > 0.5:
            self.held = False
        self.steps += 1
        success = bool(not self.held and np.linalg.norm(self.object - self.goal) < 0.09)
        return self.observe(), success or self.steps >= 50, success


class ScriptedPolicy:
    """Task-known fallback used only to test evaluation and gate wiring."""

    def act(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        action = np.zeros(ACTION_DIM)
        destination = obs["goal"] if obs["held"] else obs["object"]
        delta = destination - obs["gripper"]
        action[:2] = np.clip(delta, -0.12, 0.12)
        if not obs["held"] and np.linalg.norm(delta) < 0.055:
            action[6] = -1.0
        elif obs["held"] and np.linalg.norm(delta) < 0.055:
            action[6] = 1.0
        return action


class SimplerEnvAdapter:
    """Optional adapter for the current SimplerEnv Gym API."""

    def __init__(self, task: str):
        import simpler_env
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
        self.env = simpler_env.make(task)
        self.image_from_obs = get_image_from_maniskill2_obs_dict
        self.instruction = ""

    def _wrap(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {"agentview_image": self.image_from_obs(self.env, obs), "raw": obs}

    def reset(self, seed: int = 0) -> dict[str, Any]:
        obs, _ = self.env.reset(seed=seed)
        self.instruction = self.env.get_language_instruction()
        return self._wrap(obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], bool, bool]:
        obs, _, done, truncated, info = self.env.step(action)
        success = bool(info.get("success", done))
        return self._wrap(obs), bool(done or truncated), success

class OpenVLAPolicy:
    """Optional OpenVLA-compatible checkpoint adapter; versions belong in Appendix C."""

    def __init__(self, checkpoint: str, unnorm_key: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.torch, self.device, self.unnorm_key = torch, device, unnorm_key
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True).to(device)

    def act(self, obs: dict[str, Any], instruction: str) -> np.ndarray:
        from PIL import Image
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(obs["agentview_image"])).to(
            self.device, dtype=self.torch.bfloat16)
        return np.asarray(self.model.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False), dtype=float)


def run_episode(
    env: Any, policy: Any, seed: int, *, gate: ReversibilityGate | None = None,
    approve: Callable[[dict[str, Any]], bool] | None = None, max_steps: int = 50,
) -> dict[str, Any]:
    """Run one closed-loop episode and trust only the environment success predicate."""
    obs, latencies, success, termination = env.reset(seed), [], False, "step_limit"
    for step in range(1, max_steps + 1):
        started = time.perf_counter()
        action = np.asarray(policy.act(obs, env.instruction), dtype=float)
        latencies.append(time.perf_counter() - started)
        if gate and not gate.authorize(action, approve):
            termination = "gate_denied"
            break
        obs, done, success = env.step(action)
        if done:
            termination = "environment_success" if success else "environment_terminal"
            break
    return {"success": success, "steps": step, "termination": termination,
            "action_latency_ms": [round(value * 1000, 4) for value in latencies]}


def run_suite(env: Any, policy: Any, episodes: int = 10, **kwargs: Any) -> dict[str, Any]:
    """Aggregate final-state success, steps, latency, and gate hits."""
    rows = [run_episode(env, policy, seed, **kwargs) for seed in range(episodes)]
    latency = [value for row in rows for value in row["action_latency_ms"]]
    gate = kwargs.get("gate")
    successes = int(sum(bool(row["success"]) for row in rows))
    return {"episodes": episodes, "successes": successes,
            "success_rate": round(successes / episodes, 4),
            "mean_steps": round(float(np.mean([row["steps"] for row in rows])), 2),
            "p50_action_latency_ms": round(float(np.percentile(latency, 50)), 4),
            "p95_action_latency_ms": round(float(np.percentile(latency, 95)), 4),
            "gate_hits": gate.hits if gate else 0, "grader": "environment final-state predicate"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("toy", "simpler"), default="toy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--task", default="widowx_put_eggplant_in_basket")
    parser.add_argument("--checkpoint", default="openvla/openvla-7b")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--approve-gated", action="store_true")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.plot:
        plot_horizon(args.plot)
    env: Any = MiniTabletopEnv() if args.backend == "toy" else SimplerEnvAdapter(args.task)
    policy = ScriptedPolicy() if args.backend == "toy" else OpenVLAPolicy(args.checkpoint, args.unnorm_key)
    gate = ReversibilityGate() if args.gate else None
    approval = (lambda proposal: True) if args.approve_gated else None
    sample = np.linspace(-0.9, 0.9, ACTION_DIM)
    report = {"tokenizer": {"tokens": tokenize_action(sample).tolist(),
                             "max_roundtrip_error": roundtrip_error(sample)},
              "world_model": horizon_sweep(),
              "episode": run_suite(env, policy, args.episodes, gate=gate, approve=approval)}
    getattr(getattr(env, "env", None), "close", lambda: None)()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
