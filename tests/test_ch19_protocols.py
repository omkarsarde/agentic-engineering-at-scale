"""Focused tests for the Chapter 19 protocol seam and runtimes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


CODE = Path(__file__).parents[1] / "code" / "ch19"
sys.path.insert(0, str(CODE))
sys.modules.pop("fixture", None)

from fixture import principal, run_fixture  # noqa: E402
from protocol_lab import Principal, ProtocolError, SERVER_AUDIENCE, SupportServer  # noqa: E402
from runtimes import CheckpointLog, Client, GraphHost  # noqa: E402


def test_discovery_is_permission_filtered() -> None:
    server = SupportServer()
    reader = Principal("reader", "tenant-7", SERVER_AUDIENCE, frozenset({"policy:read"}))
    tools = server.list_tools(reader)["tools"]
    assert [tool["name"] for tool in tools] == ["policy_lookup"]


def test_wrong_audience_is_rejected() -> None:
    server = SupportServer()
    with pytest.raises(ProtocolError, match="invalid token audience"):
        server.list_tools(principal("other://resource"))


def test_protocol_version_is_explicit() -> None:
    server = SupportServer()
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        server.initialize("latest")


def test_loop_and_graph_reach_equivalent_outcomes() -> None:
    report = run_fixture()
    assert report["equivalent_outcome"] is True
    assert report["graph"]["paused"] == "awaiting_approval"
    assert report["loop"]["methods"] == report["graph"]["methods"]


def test_graph_checkpoint_is_immutable_and_forkable() -> None:
    log = CheckpointLog()
    host = GraphHost(Client(SupportServer(), principal(), "graph-host"), log)
    paused = host.start("C-42")
    fork = log.fork(-1, case_id="C-99")
    assert paused["case_id"] == "C-42"
    assert log.snapshots[-1]["case_id"] == "C-42"
    assert fork["case_id"] == "C-99"


def test_replayed_write_is_idempotent() -> None:
    server = SupportServer()
    client = Client(server, principal(), "test-host")
    client.connect()
    args = {"case_id": "C-7", "tag": "eligible", "idempotency_key": "same-effect"}
    first = client.call("case_tag", args)
    second = client.call("case_tag", args)
    assert first == second
    assert len(server.receipts) == 1
