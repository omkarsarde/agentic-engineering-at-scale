"""Optional Hugging Face generator for Chapter 8's sampler interface."""

from __future__ import annotations

import argparse
import json
import re


def extract_final_integer(text: str) -> int | None:
    """Extract the final signed integer from a candidate response."""
    matches = re.findall(r"(?<![\w.])-?\d+(?![\w.])", text.replace(",", ""))
    return int(matches[-1]) if matches else None


def generate_candidates(
    model_id: str, prompt: str, count: int = 8, seed: int = 7,
    max_new_tokens: int = 192, device: str = "cpu",
) -> list[dict[str, object]]:
    """Sample real completions without a rollout or serving engine."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install torch transformers") from exc

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    messages = [{"role": "user", "content": prompt}]
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, do_sample=True, temperature=.8, top_p=.95,
        num_return_sequences=count, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    prefix = inputs["input_ids"].shape[1]
    candidates = []
    for row in outputs:
        token_ids = row[prefix:]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        candidates.append({"text": text, "tokens": len(token_ids),
                           "answer": extract_final_integer(text)})
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("prompt")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(generate_candidates(
        args.model_id, args.prompt, args.count, args.seed,
        args.max_new_tokens, args.device,
    ), indent=2))
