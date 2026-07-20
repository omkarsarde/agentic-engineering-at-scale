"""Deterministic Chapter 12 extraction and single-tool experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chat_api import REFUND_SCHEMA, extract, parse_object, run_tool_turn
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
KNOWN_ORDERS = {f"O-{index:04d}" for index in range(1, 51)}


def make_tickets() -> list[dict[str, Any]]:
    """Create 50 source-traceable but deliberately messy support tickets."""
    tickets = []
    for index in range(1, 51):
        currency = "USD" if index % 5 else "EUR"
        amount = 900 + 37 * index
        text = (
            f"ticket T-{index:03d}; order O-{index:04d}. charged {amount / 100:.2f} "
            f"{currency}; customer asks for a refund after duplicate delivery."
        )
        tickets.append(
            {
                "ticket_id": f"T-{index:03d}",
                "order_id": f"O-{index:04d}",
                "amount_cents": amount,
                "currency": currency,
                "text": text,
            }
        )
    return tickets


def candidate(ticket: dict[str, Any], level: int) -> str:
    """Return a deterministic proxy for each structured-output guarantee."""
    index = int(ticket["ticket_id"].split("-")[1])
    value = {
        "ticket_id": ticket["ticket_id"],
        "action": "refund",
        "order_id": ticket["order_id"],
        "amount_cents": ticket["amount_cents"],
        "currency": ticket["currency"],
    }
    if level == 1 and index % 4 == 0:
        value["currency"] = ticket["currency"].lower()
    if level == 1 and index % 7 == 0:
        return "I extracted this: " + json.dumps(value) + "\nDone."
    if level == 2 and index % 11 == 0:
        value["extra"] = "not in schema"
    if level == 3 and index % 10 == 0:
        value["order_id"] = f"O-9{index:03d}"  # legal shape, nonexistent order
    return json.dumps(value)


def schema_valid(text: str) -> bool:
    """Check JSON parsing and Draft 2020-12 schema validity."""
    try:
        value = parse_object(text)
    except ValueError:
        return False
    return not any(Draft202012Validator(REFUND_SCHEMA).iter_errors(value))


class RepairInvoker:
    """One scripted failure followed by the ticket's ground-truth object."""

    def __init__(self, ticket: dict[str, Any], first: str) -> None:
        self.ticket = ticket
        self.first = first
        self.calls = 0

    def __call__(self, messages: list[dict], **_: Any) -> dict[str, Any]:
        self.calls += 1
        text = self.first if self.calls == 1 else json.dumps(
            {
                "ticket_id": self.ticket["ticket_id"],
                "action": "refund",
                "order_id": self.ticket["order_id"],
                "amount_cents": self.ticket["amount_cents"],
                "currency": self.ticket["currency"],
            }
        )
        return {
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "usage": {"total_tokens": 42 + 12 * (self.calls - 1)},
        }


class ToolInvoker:
    """Script the two API responses while preserving the real wire shape."""

    def __init__(self, order_id: str) -> None:
        self.order_id = order_id
        self.call_id = "call_" + order_id.removeprefix("O-")

    def __call__(self, messages: list[dict], **_: Any) -> dict[str, Any]:
        if messages and messages[-1].get("role") == "tool":
            assert messages[-1]["tool_call_id"] == self.call_id
            content = json.loads(messages[-1]["content"])
            return {
                "message": {"role": "assistant", "content": f"Order {self.order_id} is {content['status']}."},
                "finish_reason": "stop",
                "usage": {"total_tokens": 29},
            }
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": self.call_id,
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": self.order_id})},
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"total_tokens": 31},
        }


LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Read one order by its exact ID; do not use for product search.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": "^O-[0-9]{4}$"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


def semantic_check(value: dict[str, Any]) -> str | None:
    """Reject schema-legal identifiers that do not exist in application state."""
    return None if value["order_id"] in KNOWN_ORDERS else "order_id does not exist"


def run() -> dict[str, Any]:
    """Execute the 50-ticket extraction and 20-query tool-call build."""
    tickets = make_tickets()
    validity = {
        f"level_{level}": sum(schema_valid(candidate(ticket, level)) for ticket in tickets) / len(tickets)
        for level in (1, 2, 3)
    }
    layer3_business_invalid = sum(
        schema_valid(candidate(ticket, 3)) and semantic_check(parse_object(candidate(ticket, 3))) is not None
        for ticket in tickets
    )

    repaired = [
        extract(
            RepairInvoker(ticket, candidate(ticket, 2)),
            [{"role": "user", "content": ticket["text"]}],
            REFUND_SCHEMA,
            semantic_check=semantic_check,
        )
        for ticket in tickets
    ]
    repair_rate = sum(result.attempts > 1 for result in repaired) / len(repaired)
    tokens_per_success = sum(result.total_tokens for result in repaired) / len(repaired)

    def lookup_order(order_id: str) -> dict[str, str]:
        if order_id not in KNOWN_ORDERS:
            raise KeyError(order_id)
        return {"order_id": order_id, "status": "shipped"}

    final = []
    for index in range(1, 21):
        order_id = f"O-{index:04d}"
        response = run_tool_turn(
            ToolInvoker(order_id),
            [{"role": "user", "content": f"Where is {order_id}?"}],
            LOOKUP_TOOL,
            lookup_order,
        )
        final.append(response["finish_reason"] == "stop")

    report = {
        "tickets": len(tickets),
        "schema_validity": validity,
        "layer3_business_invalid": layer3_business_invalid,
        "repair_invocation_rate": repair_rate,
        "tokens_per_success": tokens_per_success,
        "tool_queries": len(final),
        "tool_loops_terminated": sum(final),
    }
    (ROOT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
