"""Optional Hugging Face adapter for timing and batch-invariance probes.

Install ``torch`` and ``transformers`` separately. The deterministic chapter build
never imports this module.
"""

from __future__ import annotations

import argparse
import json
import time


def _load(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.eval()
    return torch, tokenizer, model


def _tensor_bytes(value) -> int:
    """Count tensor storage recursively across legacy and modern cache objects."""
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    if hasattr(value, "to_legacy_cache"):
        value = value.to_legacy_cache()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def profile_request(model_name: str, prompt: str, max_new_tokens: int = 16, seed: int = 7) -> dict:
    """Run a manual cached greedy loop and return TTFT, TPOT, KV bytes, and text."""
    torch, tokenizer, model = _load(model_name)
    torch.manual_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt")
    token_ids, step_ms = [], []
    with torch.inference_mode():
        started = time.perf_counter()
        output = model(**encoded, use_cache=True)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ttft_ms = (time.perf_counter() - started) * 1000
        cache = output.past_key_values
        token_ids.append(int(next_token.item()))
        for _ in range(max_new_tokens - 1):
            started = time.perf_counter()
            output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_ms.append((time.perf_counter() - started) * 1000)
            cache = output.past_key_values
            token_ids.append(int(next_token.item()))
    return {"model": model_name, "prompt_tokens": int(encoded.input_ids.shape[1]),
            "output_tokens": len(token_ids), "ttft_ms": ttft_ms,
            "mean_tpot_ms": sum(step_ms) / len(step_ms) if step_ms else 0.0,
            "kv_bytes": _tensor_bytes(cache), "token_ids": token_ids,
            "text": tokenizer.decode(token_ids, skip_special_tokens=True)}


def batch_invariance(model_name: str, prompt: str, batch_size: int = 8) -> dict:
    """Compare one prompt alone with the same prompt inside an identical padded batch."""
    torch, tokenizer, model = _load(model_name)
    single = tokenizer([prompt], return_tensors="pt", padding=True)
    batched = tokenizer([prompt] * batch_size, return_tensors="pt", padding=True)
    with torch.inference_mode():
        one = model(**single).logits[0, -1].float().log_softmax(dim=-1)
        many = model(**batched).logits[0, -1].float().log_softmax(dim=-1)
    return {"model": model_name, "batch_size": batch_size,
            "max_logprob_drift": float((one - many).abs().max().item()),
            "greedy_flip": int(one.argmax().item() != many.argmax().item()),
            "single_token": int(one.argmax().item()), "batched_token": int(many.argmax().item())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps({"request": profile_request(args.model, args.prompt, args.max_new_tokens),
                      "batch_probe": batch_invariance(args.model, args.prompt, args.batch_size)},
                     indent=2, sort_keys=True))
