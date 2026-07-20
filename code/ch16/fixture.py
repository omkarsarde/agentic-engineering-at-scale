"""Deterministic strategies and tools for the Chapter 16 build."""

from __future__ import annotations

import argparse
import json
from typing import Any

from agent_loop import (
    FinalAnswer,
    Gate,
    RunState,
    ToolCall,
    ToolSpec,
    run_agent,
)


class SupportStrategy:
    """Stand in for model decisions while preserving the control boundary."""

    def __init__(self, mode: str, refund_cents: int = 4_999) -> None:
        self.mode = mode
        self.refund_cents = refund_cents

    def decide(self, state: RunState):
        if not state.observations:
            if self.mode == "router" and "policy" in state.task.lower():
                return ToolCall("c1", "read_policy", {})
            return ToolCall("c1", "lookup_order", {"order_id": "A-17"})

        last = state.observations[-1]
        if last.kind == "denied":
            if self.mode == "adaptive" and self.refund_cents > 5_000:
                self.refund_cents = 4_999
            return ToolCall(
                f"c{state.steps + 1}",
                "refund_order",
                {"order_id": "A-17", "amount_cents": self.refund_cents},
            )

        if last.call_id == "c1" and self.mode in {
            "chain",
            "router",
            "evaluator",
            "adaptive",
        }:
            return ToolCall(
                "c2",
                "refund_order",
                {"order_id": "A-17", "amount_cents": self.refund_cents},
            )
        return FinalAnswer(f"Completed with evidence: {last.content}")


def fixture() -> tuple[dict[str, ToolSpec], Gate, list[dict[str, Any]]]:
    """Create tools, policy, and an effect log for the fast path."""
    effects: list[dict[str, Any]] = []

    def lookup_order(order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "damaged", "paid_cents": 4_999}

    def read_policy() -> str:
        return "Automatic refunds are capped at 5000 cents."

    def refund_order(order_id: str, amount_cents: int) -> dict[str, Any]:
        receipt = {"order_id": order_id, "refunded_cents": amount_cents}
        effects.append(receipt)
        return receipt

    tools = {
        "lookup_order": ToolSpec({"order_id": str}, lookup_order),
        "read_policy": ToolSpec({}, read_policy),
        "refund_order": ToolSpec(
            {"order_id": str, "amount_cents": int}, refund_order, effectful=True
        ),
    }

    def gate(call: ToolCall, _: RunState) -> tuple[bool, str]:
        if call.name == "refund_order" and call.arguments["amount_cents"] > 5_000:
            return False, "refund exceeds automatic authority"
        return True, "policy permits proposal"

    return tools, gate, effects


def compare_modes() -> list[dict[str, Any]]:
    """Compare thin strategies through exactly the same execution kernel."""
    rows: list[dict[str, Any]] = []
    for mode, amount in (
        ("chain", 4_999),
        ("router", 4_999),
        ("evaluator", 4_999),
        ("adaptive", 9_999),
    ):
        tools, gate, effects = fixture()
        result = run_agent(
            "Refund damaged order A-17.",
            SupportStrategy(mode, amount),
            tools,
            gate,
        )
        rows.append(
            {
                "mode": mode,
                "stop": result.stop,
                "steps": result.state.steps,
                "cost_units": result.state.cost_units,
                "effects": len(effects),
                "task_success": effects == [
                    {"order_id": "A-17", "refunded_cents": 4_999}
                ],
            }
        )
    return rows


def main() -> None:
    """Print the deterministic comparison as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare_modes(), indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
