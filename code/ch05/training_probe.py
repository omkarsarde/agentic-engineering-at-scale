"""Paired-seed raw-versus-cleaned TinyGPT pretraining probe."""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from chapter2_adapter import BytePairTokenizer, GPTConfig, TinyGPT
from experiment_fixture import CLOZE_TASKS, FEW_SHOT_PREFIX, TOKENIZER_CALIBRATION
from training_eval import batch, bootstrap_mean_interval, candidate_score, evaluation_loss


def compare_toy_pretraining(
    raw_text: str,
    cleaned_text: str,
    holdout_text: str,
    *,
    steps: int = 80,
    seeds: tuple[int, ...] = (17, 29, 43),
) -> tuple[dict[str, object], list[dict[str, object]], str, BytePairTokenizer]:
    """Train paired Chapter 2 models with an independent frozen tokenizer."""

    if steps < 1 or not seeds:
        raise ValueError("steps and paired seeds must be non-empty and positive")
    tokenizer = BytePairTokenizer.train(TOKENIZER_CALIBRATION, vocab_size=320)
    encoded = {
        "raw": torch.tensor(tokenizer.encode(raw_text[:400_000]), dtype=torch.long),
        "cleaned": torch.tensor(tokenizer.encode(cleaned_text[:400_000]), dtype=torch.long),
    }
    holdout = torch.tensor(tokenizer.encode(holdout_text), dtype=torch.long)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2.0,
    )
    curves: list[dict[str, object]] = []
    results: dict[str, list[dict[str, float | int]]] = {"raw": [], "cleaned": []}
    sample_lines = [
        "Synthetic few-shot prompts; greedy decoding from the first paired seed is not a capability benchmark."
    ]
    parameter_count = 0
    for seed in seeds:
        torch.manual_seed(seed)
        template_model = TinyGPT(config)
        template = {key: value.clone() for key, value in template_model.state_dict().items()}
        parameter_count = template_model.parameter_count()
        for condition in ("raw", "cleaned"):
            model = TinyGPT(config)
            model.load_state_dict(template)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
            generator = torch.Generator().manual_seed(seed + 1)
            model.train()
            for step in range(steps):
                inputs, targets = batch(encoded[condition], config.block_size, 8, generator)
                _, loss, _ = model(inputs, targets)
                assert loss is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                curves.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "step": step + 1,
                        "training_loss": float(loss.detach()),
                    }
                )
            evaluation = evaluation_loss(model, holdout)
            correct = 0
            for question, candidates, answer in CLOZE_TASKS:
                prompt = FEW_SHOT_PREFIX + question
                scores = [candidate_score(model, tokenizer, prompt, item) for item in candidates]
                correct += int(int(np.argmax(scores)) == answer)
                if seed == seeds[0]:
                    prompt_ids = tokenizer.encode(prompt)[-(config.block_size - 12) :]
                    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
                    generated = model.generate(prompt_tensor, max_new_tokens=12, temperature=0.0)
                    sample_lines.append(
                        f"[{condition}; seed={seed}] "
                        f"{tokenizer.decode(generated[0].tolist(), errors='replace')}"
                    )
            results[condition].append(
                {
                    "seed": seed,
                    "heldout_loss": evaluation,
                    "cloze_accuracy": correct / len(CLOZE_TASKS),
                }
            )

    answer_positions = [answer for _, _, answer in CLOZE_TASKS]
    shortest_correct = sum(
        int(
            min(
                range(len(candidates)),
                key=lambda index: (len(tokenizer.encode(candidates[index])), index),
            )
            == answer
        )
        for _, candidates, answer in CLOZE_TASKS
    )
    summary: dict[str, object] = {
        "steps": steps,
        "paired_seeds": list(seeds),
        "parameter_count": parameter_count,
        "configuration": {
            "tokenizer_calibration": "independent synthetic calibration text",
            "vocab_size": tokenizer.vocab_size,
            "block_size": config.block_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "batch_size": 8,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.01,
        },
        "cloze_contract": {
            "few_shot_examples": 2,
            "correct_answer_positions": answer_positions,
            "constant_position_baseline_accuracy": (
                max(Counter(answer_positions).values()) / len(CLOZE_TASKS)
            ),
            "shortest_candidate_baseline_accuracy": shortest_correct / len(CLOZE_TASKS),
            "length_normalization": "mean completion log probability",
        },
    }
    for offset, condition in enumerate(("raw", "cleaned")):
        losses = [float(row["heldout_loss"]) for row in results[condition]]
        accuracies = [float(row["cloze_accuracy"]) for row in results[condition]]
        low, high = bootstrap_mean_interval(losses, seed=700 + offset)
        summary[condition] = {
            "heldout_loss": float(np.mean(losses)),
            "heldout_loss_p05": low,
            "heldout_loss_p95": high,
            "heldout_loss_by_seed": results[condition],
            "heldout_perplexity": float(np.exp(np.mean(losses))),
            "cloze_accuracy": float(np.mean(accuracies)),
            "training_tokens_available": int(encoded[condition].numel()),
        }
    paired = [
        float(raw["heldout_loss"]) - float(cleaned["heldout_loss"])
        for raw, cleaned in zip(results["raw"], results["cleaned"])
    ]
    low, high = bootstrap_mean_interval(paired, seed=811)
    summary["paired_raw_minus_cleaned"] = {
        "mean": float(np.mean(paired)),
        "p05": low,
        "p95": high,
        "by_seed": paired,
        "cleaned_wins_every_seed": all(value > 0 for value in paired),
    }
    return summary, curves, "\n".join(sample_lines) + "\n", tokenizer
