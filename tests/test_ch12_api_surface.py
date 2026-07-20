"""Focused tests for Chapter 12's owned API boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


CODE = Path(__file__).parents[1] / "code" / "ch12"
sys.path.insert(0, str(CODE))
sys.modules.pop("chapter_build", None)

from chapter_build import LOOKUP_TOOL, RepairInvoker, candidate, make_tickets, run, semantic_check  # noqa: E402
from chat_api import (  # noqa: E402
    REFUND_SCHEMA,
    IncompleteToolStream,
    StructuredOutputFailed,
    assemble_tool_stream,
    assert_all_calls_answered,
    chat,
    extract,
    run_tool_turn,
)


def test_raw_chat_posts_the_expected_openai_compatible_shape() -> None:
    seen = {}

    def transport(url: str, body: dict) -> dict:
        seen.update(url=url, body=body)
        return {"id": "req_1", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"total_tokens": 8}}

    result = chat("http://localhost:8000", "local-model", [{"role": "user", "content": "hi"}], transport=transport)
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["messages"][0]["role"] == "user"
    assert result["finish_reason"] == "stop"


def test_bounded_repair_reaches_application_validity() -> None:
    ticket = make_tickets()[10]
    result = extract(
        RepairInvoker(ticket, candidate(ticket, 2)),
        [{"role": "user", "content": ticket["text"]}],
        REFUND_SCHEMA,
        semantic_check=semantic_check,
    )
    assert result.attempts == 2
    assert result.value["order_id"] == ticket["order_id"]


def test_bounded_repair_returns_a_typed_failure() -> None:
    def hopeless(messages: list[dict], **kwargs: object) -> dict:
        return {"message": {"role": "assistant", "content": "not json"}, "finish_reason": "stop", "usage": {"total_tokens": 5}}

    with pytest.raises(StructuredOutputFailed):
        extract(hopeless, [{"role": "user", "content": "extract"}], REFUND_SCHEMA, max_attempts=2)


def test_schema_validity_does_not_prove_order_existence() -> None:
    report = run()
    assert report["schema_validity"]["level_3"] == 1.0
    assert report["layer3_business_invalid"] == 5


def test_every_call_id_must_be_answered_exactly_once() -> None:
    calls = [{"id": "call_1"}, {"id": "call_2"}]
    with pytest.raises(ValueError):
        assert_all_calls_answered(calls, [{"role": "tool", "tool_call_id": "call_1"}])
    with pytest.raises(ValueError):
        assert_all_calls_answered(calls, [{"role": "tool", "tool_call_id": "call_1"}, {"role": "tool", "tool_call_id": "call_1"}])


def test_single_tool_turn_correlates_the_result() -> None:
    state = {"turn": 0}

    def invoke(messages: list[dict], **kwargs: object) -> dict:
        state["turn"] += 1
        if state["turn"] == 1:
            return {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "O-0009"})}}]}, "finish_reason": "tool_calls"}
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "call_9"
        return {"message": {"role": "assistant", "content": "shipped"}, "finish_reason": "stop"}

    result = run_tool_turn(invoke, [{"role": "user", "content": "status?"}], LOOKUP_TOOL, lambda order_id: {"order_id": order_id, "status": "shipped"})
    assert result["finish_reason"] == "stop"


def test_invalid_tool_arguments_never_reach_the_handler() -> None:
    called = False

    def invoke(messages: list[dict], **kwargs: object) -> dict:
        return {"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call_bad", "type": "function", "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": 9})}}]}, "finish_reason": "tool_calls"}

    def handler(order_id: str) -> dict:
        nonlocal called
        called = True
        return {"order_id": order_id}

    with pytest.raises(ValueError, match="tool arguments failed validation"):
        run_tool_turn(invoke, [{"role": "user", "content": "status?"}], LOOKUP_TOOL, handler)
    assert not called


def test_stream_is_not_actionable_without_terminal_event() -> None:
    chunks = [
        {"tool_call": {"index": 0, "id": "call_1", "name": "lookup_order", "arguments": '{"order_'}},
        {"tool_call": {"index": 0, "arguments": 'id":"O-0001"}', "closed": True}},
    ]
    with pytest.raises(IncompleteToolStream):
        assemble_tool_stream(chunks)
    complete = assemble_tool_stream([*chunks, {"terminal": True}])
    assert complete[0]["arguments"] == {"order_id": "O-0001"}


def test_build_finishes_every_tool_loop_and_measures_repair_cost() -> None:
    report = run()
    assert report["tool_loops_terminated"] == report["tool_queries"] == 20
    assert report["schema_validity"]["level_1"] < report["schema_validity"]["level_2"] < report["schema_validity"]["level_3"]
    assert report["repair_invocation_rate"] > 0
    assert report["tokens_per_success"] > 42
