"""Run the Chapter 19 protocol and framework comparison."""

from __future__ import annotations

import json

from protocol_lab import Principal, ProtocolError, SERVER_AUDIENCE, SupportServer
from runtimes import CheckpointLog, Client, GraphHost, LoopHost


def principal(audience: str = SERVER_AUDIENCE) -> Principal:
    return Principal(
        "engineer-4",
        "tenant-7",
        audience,
        frozenset({"policy:read", "case:write"}),
    )


def run_fixture() -> dict:
    loop = LoopHost(Client(SupportServer(), principal(), "loop-host"))
    loop_result = loop.run("C-41")

    checkpoints = CheckpointLog()
    graph = GraphHost(Client(SupportServer(), principal(), "graph-host"), checkpoints)
    paused = graph.start("C-41")
    graph_result = graph.resume(paused, approved=True)

    hostile = SupportServer()
    try:
        hostile.list_tools(principal("inventory://api"))
        wrong_audience = "unexpectedly accepted"
    except ProtocolError as error:
        wrong_audience = str(error)

    return {
        "protocol": {
            "version": "2025-11-25",
            "wrong_audience": wrong_audience,
        },
        "loop": loop_result,
        "graph": {
            "paused": paused["status"],
            "status": graph_result["status"],
            "answer": graph_result["answer"],
            "methods": graph_result["methods"],
            "checkpoints": len(checkpoints.snapshots),
        },
        "equivalent_outcome": loop_result["answer"] == graph_result["answer"],
    }


if __name__ == "__main__":
    print(json.dumps(run_fixture(), indent=2))
