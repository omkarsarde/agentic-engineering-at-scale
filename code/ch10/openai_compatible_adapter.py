"""Optional network adapter for timing an OpenAI-compatible streaming endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request


def profile_request(base_url: str, model: str, prompt: str, *, api_key: str = "",
                    max_tokens: int = 64, timeout: float = 120.0) -> dict[str, object]:
    """Measure first-content latency and stream gaps without third-party packages."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started, content_times, pieces, usage = time.perf_counter(), [], [], {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            content = delta.get("content")
            if content:
                content_times.append(time.perf_counter()); pieces.append(content)
    ended = time.perf_counter()
    gaps = [later - earlier for earlier, later in zip(content_times, content_times[1:])]
    return {"model": model, "ttfc_seconds": content_times[0] - started if content_times else None,
            "total_seconds": ended - started, "content_events": len(content_times),
            "mean_content_event_gap_seconds": statistics.fmean(gaps) if gaps else None,
            "output_tokens_reported": usage.get("completion_tokens"), "text": "".join(pieces)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("prompt")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    arguments = parser.parse_args()
    print(json.dumps(profile_request(arguments.base_url, arguments.model, arguments.prompt,
                                     api_key=arguments.api_key,
                                     max_tokens=arguments.max_tokens), indent=2))
