"""Fast-path experiments for the Chapter 17 harness."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from harness import (
    ApprovalError,
    Call,
    ExecutionContext,
    Risk,
    Tool,
    approve,
    dispatch,
    request_approval,
    select_tools,
)
from state import Journal


def make_tools(state: dict[str, Any]) -> list[Tool]:
    """Create a six-tool surface around one mutable order fixture."""
    def lookup_order(order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, **state}

    def read_policy(topic: str) -> str:
        return f"Policy for {topic}: auto-refund limit is 5000 cents."

    def read_customer(customer_id: str) -> dict[str, str]:
        return {"customer_id": customer_id, "tier": "standard"}

    def propose_refund(order_id: str, amount_cents: int) -> dict[str, Any]:
        return {"order_id": order_id, "proposed_cents": amount_cents}

    def issue_refund(order_id: str, amount_cents: int) -> dict[str, Any]:
        state["refunded_cents"] = amount_cents
        state["version"] = f"v{int(state['version'][1:]) + 1}"
        return {"order_id": order_id, "refunded_cents": amount_cents}

    def add_case_note(order_id: str, note: str) -> dict[str, str]:
        return {"order_id": order_id, "note": note}

    def version(_: dict[str, Any]) -> str:
        return state["version"]

    return [
        Tool("order_lookup", "read order status and payment", {"order_id": str}, Risk.READ, lookup_order),
        Tool("policy_read", "read refund policy", {"topic": str}, Risk.READ, read_policy),
        Tool("customer_read", "read customer tier", {"customer_id": str}, Risk.READ, read_customer),
        Tool("refund_propose", "draft refund without effect", {"order_id": str, "amount_cents": int}, Risk.READ, propose_refund),
        Tool("refund_issue", "issue approved refund payment", {"order_id": str, "amount_cents": int}, Risk.WRITE, issue_refund, version),
        Tool("case_note_add", "write a support case note", {"order_id": str, "note": str}, Risk.WRITE, add_case_note, version),
    ]


def run_attack() -> dict[str, Any]:
    """Approve one action, mutate it and its target, and verify containment."""
    state: dict[str, Any] = {"status": "damaged", "version": "v1"}
    tools = {tool.name: tool for tool in make_tools(state)}
    context = ExecutionContext("tenant-7", "support-agent")
    original = Call("c1", "refund_issue", {"order_id": "A-17", "amount_cents": 4_999})
    request = request_approval(original, context, state["version"], ttl_s=60, now=100)
    token = approve(request, "manager-3")

    substituted = Call("c1", "refund_issue", {"order_id": "A-17", "amount_cents": 9_999})
    outcomes: dict[str, Any] = {}
    for name, call, mutate in (
        ("substituted", substituted, False),
        ("stale", original, True),
    ):
        if mutate:
            state["version"] = "v2"
        try:
            dispatch(call, context, tools, token, now=120)
            outcomes[name] = "escaped"
        except ApprovalError as exc:
            outcomes[name] = str(exc)

    state["version"] = "v3"
    fresh = approve(
        request_approval(original, context, state["version"], ttl_s=60, now=120),
        "manager-3",
    )
    outcomes["fresh"] = dispatch(original, context, tools, fresh, now=121)
    return outcomes


def run_resume() -> dict[str, Any]:
    """Replay a harness event after restart without duplicating the audit row."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "harness.sqlite"
        first = Journal(path)
        first.record("evt-approval-A17", "thread-1", "approved", {"amount": 4_999})
        first.checkpoint("thread-1", {"phase": "approved", "step": 2})
        first.db.close()

        resumed = Journal(path)
        inserted = resumed.record(
            "evt-approval-A17", "thread-1", "approved", {"amount": 4_999}
        )
        return {
            "checkpoint": resumed.load("thread-1"),
            "duplicate_inserted": inserted,
            "audit_rows": resumed.event_count("thread-1"),
        }


def run_surface_eval() -> dict[str, Any]:
    """Measure retrieval accuracy and exposed schema characters."""
    tools = make_tools({"status": "damaged", "version": "v1"})
    cases = [
        ("look up order status", "order_lookup"),
        ("read refund policy", "policy_read"),
        ("read customer tier", "customer_read"),
        ("draft a refund proposal", "refund_propose"),
        ("issue approved refund", "refund_issue"),
        ("add support case note", "case_note_add"),
    ]
    hits = 0
    exposed = 0
    for query, expected in cases:
        selected = select_tools(query, tools, limit=2)
        hits += expected in {tool.name for tool in selected}
        exposed += sum(len(tool.name) + len(tool.summary) for tool in selected)
    preload = sum(len(tool.name) + len(tool.summary) for tool in tools) * len(cases)
    return {
        "recall_at_2": hits / len(cases),
        "retrieved_schema_chars": exposed,
        "preloaded_schema_chars": preload,
    }


def plot_surface(metrics: dict[str, Any], path: Path) -> None:
    """Render the measured context-surface comparison as SVG."""
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "chapter-17"
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Preload all six", "Retrieve top two"]
    values = [metrics["preloaded_schema_chars"], metrics["retrieved_schema_chars"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(labels, values, color=["#9a6b1f", "#315b8a"])
    ax.bar_label(bars, labels=[f"{value:,} chars" for value in values], padding=3)
    ax.set_ylabel("Tool name + summary characters across six tasks")
    ax.set_title("Deferred loading preserved recall@2 = 1.0 in the fixture")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    surface = run_surface_eval()
    if args.plot:
        plot_surface(surface, args.plot)
    print(json.dumps({"surface": surface, "attack": run_attack(), "resume": run_resume()}, indent=2))
