"""Probe sampling, token log probabilities, and serving latency.

The fixture client is deterministic and offline.  The OpenAI-compatible client
is an optional thin adapter for a locally served model, such as vLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx


@dataclass(frozen=True)
class TokenScore:
    """A returned token and its log probability under the serving model."""

    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """The small, provider-neutral result used by this chapter's probe."""

    text: str
    token_scores: tuple[TokenScore, ...]
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float


class CompletionClient(Protocol):
    """The one operation needed by the probe."""

    def complete(self, prompt: str, temperature: float, seed: int) -> Completion: ...


class FixtureClient:
    """Offline model-shaped fixture with reproducible sampling behavior."""

    _answers = (
        "Run the smallest evaluation before release.",
        "Check one representative failure before release.",
        "Measure the behavior you plan to ship.",
        "Release only after a bounded smoke test.",
    )
    _logits = (1.6, 0.9, 0.3, -0.4)

    def complete(self, prompt: str, temperature: float, seed: int) -> Completion:
        if "Alpine Berry Treaty" in prompt:
            text = (
                "The 1912 Alpine Berry Treaty created seasonal berry quotas "
                "for the mountain cantons and a joint inspection council."
            )
            scores = tuple(TokenScore(t, -0.03 - i * 0.002) for i, t in enumerate(text.split()))
        else:
            index = 0 if temperature == 0 else self._sample(temperature, seed)
            text = self._answers[index]
            scores = tuple(TokenScore(t, -0.08 - i * 0.01) for i, t in enumerate(text.split()))
        output_tokens = len(scores)
        # Synthetic timing keeps the offline lab stable; it is not a benchmark.
        ttft_ms = 12.0 + len(prompt.split()) * 0.4
        total_ms = ttft_ms + max(output_tokens - 1, 0) * 2.5
        return Completion(text, scores, len(prompt.split()), output_tokens, ttft_ms, total_ms)

    def _sample(self, temperature: float, seed: int) -> int:
        scaled = [value / temperature for value in self._logits]
        peak = max(scaled)
        weights = [math.exp(value - peak) for value in scaled]
        draw = random.Random(seed).random() * sum(weights)
        for index, weight in enumerate(weights):
            draw -= weight
            if draw <= 0:
                return index
        return len(weights) - 1


class OpenAICompatibleClient:
    """Thin streaming adapter for an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY") -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, prompt: str, temperature: float, seed: int) -> Completion:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": 48,
            "logprobs": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        scores: list[TokenScore] = []
        usage: dict[str, int] = {}
        with httpx.stream("POST", self.url, headers=self.headers, json=payload, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece and first_token_at is None:
                    first_token_at = time.perf_counter()
                text_parts.append(piece)
                for item in (choices[0].get("logprobs") or {}).get("content") or []:
                    scores.append(TokenScore(item["token"], float(item["logprob"])))
        finished = time.perf_counter()
        first_token_at = first_token_at or finished
        return Completion(
            "".join(text_parts),
            tuple(scores),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", len(scores))),
            (first_token_at - started) * 1_000,
            (finished - started) * 1_000,
        )


MEMO = """# Agency-budget memo

## Proposed outcome
- Task and measurable success condition:
- People or systems affected:

## Least-autonomous baseline
- Rule, query, search, solver, workflow, or human baseline:
- Evidence that the baseline is insufficient:

## Authority requested
| Action | Read/write | Scope | Approval | Reversal |
|---|---|---|---|---|
| Example action | read | one record | none | n/a |

## Incremental budget
- Expected quality or throughput gain:
- Token and compute budget:
- Latency and variance budget:
- Evaluation and red-team budget:
- Security, review, and operator-attention budget:
- Maximum blast radius:

## Release contract
- Offline evidence required:
- Online canary and monitor:
- Escalation, stop, and rollback conditions:
- Owner and review date:
"""


def run_probe(client: CompletionClient, out_dir: Path, seeds: range = range(12)) -> dict[float, int]:
    """Run the chapter experiment and write its inspectable artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    unique: dict[float, int] = {}
    prompt = "Give one short release-readiness reminder."
    for temperature in (0.0, 0.7, 1.2):
        completions = [client.complete(prompt, temperature, seed) for seed in seeds]
        unique[temperature] = len({item.text for item in completions})
        for seed, item in zip(seeds, completions):
            item_row = asdict(item)
            item_row["scored_tokens"] = len(item_row.pop("token_scores"))
            rows.append({"temperature": temperature, "seed": seed, **item_row})
    with (out_dir / "sampling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    false_premise = client.complete("What did the 1912 Alpine Berry Treaty establish?", 0.0, 0)
    report = {
        "prompt": "What did the 1912 Alpine Berry Treaty establish?",
        "completion": false_premise.text,
        "token_scores": [asdict(item) for item in false_premise.token_scores],
        "warning": "Token log probabilities are not probabilities that the factual claim is true.",
    }
    (out_dir / "confabulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "agency-budget-template.md").write_text(MEMO, encoding="utf-8")
    _plot(unique, out_dir / "sampling-diversity.svg")
    return unique


def _plot(unique: dict[float, int], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-01"
    temperatures = list(unique)
    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    axis.bar([str(value) for value in temperatures], unique.values(), color="#2a7f9e")
    axis.set(xlabel="Temperature", ylabel="Distinct outputs in 12 seeded calls", ylim=(0, 4.5))
    axis.bar_label(axis.containers[0])
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixture", "openai-compatible"), default="fixture")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    if args.mode == "openai-compatible" and not args.model:
        parser.error("--model is required with --mode openai-compatible")
    client: CompletionClient = (
        OpenAICompatibleClient(args.base_url, args.model, os.getenv("VLLM_API_KEY", "EMPTY"))
        if args.mode == "openai-compatible"
        else FixtureClient()
    )
    print(json.dumps(run_probe(client, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
