"""Two control-flow paradigms over the same Chapter 19 protocol client."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from protocol_lab import PROTOCOL_VERSION, Principal, SupportServer


@dataclass
class Client:
    """One host-owned client connection to exactly one server."""

    server: SupportServer
    principal: Principal
    host_name: str
    sequence: int = 0
    transcript: list[str] = field(default_factory=list)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.sequence += 1
        response = self.server.dispatch(
            {"jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params or {}},
            self.principal,
        )
        if response["id"] != self.sequence:
            raise RuntimeError("response correlation failed")
        self.transcript.append(method)
        return response["result"]

    def connect(self) -> list[str]:
        self.request("initialize", {"protocolVersion": PROTOCOL_VERSION})
        listed = self.request("tools/list")
        return [tool["name"] for tool in listed["tools"]]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})[
            "structuredContent"
        ]


def _tag_arguments(case_id: str) -> dict[str, str]:
    return {"case_id": case_id, "tag": "manual_review", "idempotency_key": f"tag:{case_id}:v3"}


class LoopHost:
    """A model-driven loop whose deterministic planner is a chapter fixture."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def run(self, case_id: str) -> dict[str, Any]:
        tools = self.client.connect()
        observations: list[dict[str, Any]] = []
        for proposal in (
            ("policy_lookup", {"topic": "refund"}),
            ("case_tag", _tag_arguments(case_id)),
        ):
            if proposal[0] not in tools:
                raise RuntimeError(f"tool unavailable: {proposal[0]}")
            observations.append(self.client.call(*proposal))
        return {
            "status": "completed",
            "answer": f"{case_id} routed to {observations[-1]['tag']} under policy v3",
            "methods": tuple(self.client.transcript),
        }


@dataclass
class CheckpointLog:
    """Append immutable state snapshots; this is not durable execution."""

    snapshots: list[dict[str, Any]] = field(default_factory=list)

    def save(self, state: dict[str, Any]) -> None:
        self.snapshots.append(deepcopy(state))

    def fork(self, index: int, **updates: Any) -> dict[str, Any]:
        state = deepcopy(self.snapshots[index])
        state.update(updates)
        return state


class GraphHost:
    """An explicit state graph with a review interrupt and checkpoints."""

    def __init__(self, client: Client, checkpoints: CheckpointLog | None = None) -> None:
        self.client = client
        self.checkpoints = checkpoints or CheckpointLog()

    def start(self, case_id: str) -> dict[str, Any]:
        state: dict[str, Any] = {"node": "discover", "case_id": case_id, "status": "running"}
        tools = self.client.connect()
        state.update(node="lookup", tools=tools)
        self.checkpoints.save(state)
        policy = self.client.call("policy_lookup", {"topic": "refund"})
        state.update(node="review", policy=policy, status="awaiting_approval")
        self.checkpoints.save(state)
        return deepcopy(state)

    def resume(self, state: dict[str, Any], approved: bool) -> dict[str, Any]:
        if state.get("node") != "review" or state.get("status") != "awaiting_approval":
            raise RuntimeError("state is not at the review interrupt")
        resumed = deepcopy(state)
        if not approved:
            resumed.update(node="finish", status="rejected", answer="reviewer rejected routing")
            self.checkpoints.save(resumed)
            return resumed
        receipt = self.client.call("case_tag", _tag_arguments(resumed["case_id"]))
        resumed.update(
            node="finish",
            status="completed",
            receipt=receipt,
            answer=f"{resumed['case_id']} routed to {receipt['tag']} under policy v3",
            methods=tuple(self.client.transcript),
        )
        self.checkpoints.save(resumed)
        return resumed
