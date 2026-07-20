"""Optional Hugging Face adapter for real sequence log-probability inspection.

Install ``torch`` and ``transformers`` and pass an instruct/base model that has a
chat template. The deterministic chapter build does not import this module.
"""

from __future__ import annotations

import argparse
import json


def sequence_logps(
    model_id: str,
    messages: list[dict[str, str]],
    candidates: list[str],
    device: str = "cpu",
) -> list[float]:
    """Return summed completion-token log-probabilities for real candidates."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install torch transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    if tokenizer.chat_template is None:
        raise ValueError(f"{model_id!r} has no tokenizer chat_template")
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    scores = []
    with torch.no_grad():
        for candidate in candidates:
            full_ids = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": candidate}],
                tokenize=True, add_generation_prompt=False,
            )
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("chat template has no stable assistant-generation prefix")
            input_ids = torch.tensor([full_ids], device=device)
            logits = model(input_ids=input_ids).logits[:, :-1].log_softmax(-1)
            labels = input_ids[:, 1:]
            token_logps = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            start = len(prompt_ids) - 1
            scores.append(float(token_logps[:, start:].sum().cpu()))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--prompt", default="Explain why the sky appears blue in one sentence.")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    values = sequence_logps(
        args.model_id, [{"role": "user", "content": args.prompt}],
        args.candidate, args.device,
    )
    print(json.dumps(dict(zip(args.candidate, values)), indent=2))
