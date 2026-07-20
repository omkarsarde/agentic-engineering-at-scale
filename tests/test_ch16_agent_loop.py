"""Executable invariants for the Chapter 16 agent kernel."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch16"))
sys.modules.pop("fixture", None)

from agent_loop import (  # noqa: E402
    FinalAnswer,
    Limits,
    Stop,
    ToolCall,
    run_agent,
)
from fixture import SupportStrategy, fixture  # noqa: E402


def test_denied_effect_never_reaches_handler() -> None:
    tools, gate, effects = fixture()
    result = run_agent(
        "Refund A-17",
        SupportStrategy("chain", refund_cents=9_999),
        tools,
        gate,
        Limits(max_steps=5, repeated_denials=1),
    )
    assert result.stop == Stop.NO_PROGRESS
    assert effects == []


def test_adaptive_strategy_uses_denial_as_observation() -> None:
    tools, gate, effects = fixture()
    result = run_agent(
        "Refund A-17",
        SupportStrategy("adaptive", refund_cents=9_999),
        tools,
        gate,
    )
    assert result.stop == Stop.ANSWERED
    assert effects == [{"order_id": "A-17", "refunded_cents": 4_999}]
    assert any(item.kind == "denied" for item in result.state.observations)


def test_step_budget_terminates_a_wanderer() -> None:
    class Wanderer:
        def decide(self, state):
            return ToolCall(f"c{state.steps}", "lookup_order", {"order_id": "A-17"})

    tools, gate, _ = fixture()
    result = run_agent("Keep looking", Wanderer(), tools, gate, Limits(max_steps=3))
    assert result.stop == Stop.STEP_LIMIT
    assert result.state.steps == 3


def test_final_answer_has_typed_success() -> None:
    class Immediate:
        def decide(self, _state):
            return FinalAnswer("done")

    tools, gate, effects = fixture()
    result = run_agent("Answer", Immediate(), tools, gate)
    assert result.stop == Stop.ANSWERED
    assert result.answer == "done"
    assert effects == []
