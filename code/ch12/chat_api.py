"""Raw chat-completions boundary with schema and single-tool guarantees."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


Message = dict[str, Any]
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]
Invoker = Callable[..., dict[str, Any]]

REFUND_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string", "pattern": "^T-[0-9]{3}$"},
        "action": {"enum": ["refund", "deny", "escalate"]},
        "order_id": {"type": "string", "pattern": "^O-[0-9]{4}$"},
        "amount_cents": {"type": "integer", "minimum": 0},
        "currency": {"enum": ["USD", "EUR"]},
    },
    "required": ["ticket_id", "action", "order_id", "amount_cents", "currency"],
    "additionalProperties": False,
}


class StructuredOutputFailed(RuntimeError):
    """No valid application object was obtained within the attempt budget."""


class IncompleteToolStream(RuntimeError):
    """A streamed tool proposal never reached both close and terminal events."""


@dataclass(frozen=True)
class Extraction:
    """Validated object plus the cost of obtaining it."""

    value: dict[str, Any]
    attempts: int
    total_tokens: int


def _http_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON without importing a provider SDK."""
    import httpx

    response = httpx.post(url, json=body, timeout=30.0)
    response.raise_for_status()
    return response.json()


def chat(
    base_url: str,
    model: str,
    messages: list[Message],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
    tool_choice: Any = None,
    parallel_tool_calls: bool = False,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat-completions endpoint over raw HTTP."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        body["tools"] = tools
        body["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        body["response_format"] = response_format
    if tool_choice is not None:
        body["tool_choice"] = tool_choice

    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    wire = (transport or _http_transport)(endpoint, body)
    choice = wire["choices"][0]
    return {
        "message": choice["message"],
        "finish_reason": choice.get("finish_reason"),
        "usage": wire.get("usage", {}),
        "request_id": wire.get("id"),
    }


def count_tokens(tokenizer: Any, messages: list[Message], tools: list[dict] | None = None) -> int:
    """Count the rendered chat with the exact tokenizer and template."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Create the OpenAI-compatible strict JSON-Schema request shape."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def parse_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object, allowing prose only for prompt-only mode."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("the first JSON value is not an object")
    return value


def _complaint(
    value: dict[str, Any],
    validator: Draft202012Validator,
    semantic_check: Callable[[dict[str, Any]], str | None] | None,
) -> str | None:
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        return "; ".join(
            f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors
        )
    return semantic_check(value) if semantic_check else None


def extract(
    invoke: Invoker,
    messages: list[Message],
    schema: dict[str, Any],
    *,
    max_attempts: int = 3,
    semantic_check: Callable[[dict[str, Any]], str | None] | None = None,
) -> Extraction:
    """Parse, validate, repair with a bounded budget, or return typed failure."""
    validator = Draft202012Validator(schema)
    transcript = list(messages)
    tokens = 0
    for attempt in range(1, max_attempts + 1):
        result = invoke(
            transcript,
            response_format=schema_response_format("refund_decision", schema),
        )
        tokens += int(result.get("usage", {}).get("total_tokens", 0))
        text = result["message"].get("content") or ""
        try:
            value = parse_object(text)
            complaint = _complaint(value, validator, semantic_check)
        except ValueError as exc:
            complaint = str(exc)
        if complaint is None:
            return Extraction(value=value, attempts=attempt, total_tokens=tokens)
        transcript.extend(
            [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"The candidate failed validation: {complaint}. "
                        "Return only one object matching the supplied schema."
                    ),
                },
            ]
        )
    raise StructuredOutputFailed(f"no valid object after {max_attempts} attempts")


def assert_all_calls_answered(calls: list[dict], results: list[Message]) -> None:
    """Require every proposal ID to have exactly one result ID."""
    proposed = [call["id"] for call in calls]
    answered = [result["tool_call_id"] for result in results]
    if len(proposed) != len(set(proposed)) or sorted(proposed) != sorted(answered):
        raise ValueError("tool results must answer every unique call id exactly once")


def run_tool_turn(
    invoke: Invoker,
    messages: list[Message],
    tool: dict[str, Any],
    handler: Callable[..., Any],
) -> dict[str, Any]:
    """Run one proposed tool call and return the model's final response."""
    first = invoke(messages, tools=[tool], parallel_tool_calls=False)
    if first["finish_reason"] != "tool_calls":
        raise ValueError("expected a tool_calls finish reason")
    calls = first["message"].get("tool_calls", [])
    if len(calls) != 1:
        raise ValueError("this boundary permits exactly one tool proposal")
    call = calls[0]
    if call["function"]["name"] != tool["function"]["name"]:
        raise ValueError("the proposed tool is not allowed")
    arguments = json.loads(call["function"]["arguments"])
    argument_errors = list(
        Draft202012Validator(tool["function"]["parameters"]).iter_errors(arguments)
    )
    if argument_errors:
        raise ValueError(f"tool arguments failed validation: {argument_errors[0].message}")
    output = handler(**arguments)
    result_message = {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(output, sort_keys=True),
    }
    assert_all_calls_answered(calls, [result_message])
    transcript = [*messages, first["message"], result_message]
    return invoke(transcript, tools=[tool], parallel_tool_calls=False)


def assemble_tool_stream(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble arguments, but expose proposals only after response termination."""
    buffers: dict[int, dict[str, Any]] = {}
    terminal = False
    for chunk in chunks:
        if chunk.get("terminal"):
            terminal = True
            continue
        delta = chunk["tool_call"]
        index = int(delta["index"])
        state = buffers.setdefault(index, {"id": None, "name": None, "text": "", "closed": False})
        state["id"] = delta.get("id", state["id"])
        state["name"] = delta.get("name", state["name"])
        state["text"] += delta.get("arguments", "")
        state["closed"] = state["closed"] or bool(delta.get("closed"))
    if not terminal or not buffers or any(not state["closed"] for state in buffers.values()):
        raise IncompleteToolStream("stream ended before call-close and response-terminal events")
    return [
        {"id": state["id"], "name": state["name"], "arguments": json.loads(state["text"])}
        for _, state in sorted(buffers.items())
    ]


def classify_failure(*, status: int | None = None, finish_reason: str | None = None, error: Exception | None = None) -> str:
    """Map a boundary symptom to the layer that owns recovery."""
    if error is not None:
        return "code_retry" if error.__class__.__name__ in {"ConnectError", "TimeoutException"} else "typed_failure"
    if status is not None and status >= 400:
        return "code_retry" if status in {408, 429, 500, 502, 503, 504} else "typed_failure"
    if finish_reason == "length":
        return "model_retry_without_effect"
    if finish_reason == "content_filter":
        return "typed_refusal"
    return "accept"
