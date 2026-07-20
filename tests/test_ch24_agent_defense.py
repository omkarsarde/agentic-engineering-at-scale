"""Focused tests for the Chapter 24 containment boundary."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch24"
sys.path.insert(0, str(CODE))
sys.modules.pop("fixture", None)

from agent_defense import (  # noqa: E402
    Action,
    Approval,
    AuditLog,
    EnforcementPoint,
    POLICY_VERSION,
    PolicyEngine,
    Principal,
    action_digest,
)
from fixture import run_fixture  # noqa: E402


def principal() -> Principal:
    return Principal(
        "agent-runtime",
        "tenant-7",
        frozenset({"refund:write", "treasury:write", "network:fetch"}),
    )


def world() -> dict:
    return {"balance": 1000, "refunds": [], "fetched": []}


def test_naive_is_exploited_and_gate_contains() -> None:
    report = run_fixture()
    assert report["naive"]["attack_success_rate"] == 1.0
    assert report["tool-gate"]["attack_success_rate"] == 0.0
    assert report["full"]["attack_success_rate"] == 0.0


def test_detector_miss_does_not_break_containment() -> None:
    report = run_fixture()
    assert report["detector-only"]["attack_success_rate"] > 0
    assert report["tool-gate"]["containment_rate"] == 1.0


def test_irreversible_action_requires_external_approval() -> None:
    action = Action("refund", {"order_id": "A-17"}, irreversible=True)
    state, audit = world(), AuditLog()
    decision = EnforcementPoint(PolicyEngine(), audit).execute(action, principal(), state)
    assert decision.effect == "review"
    assert state["refunds"] == []


def test_bound_approval_allows_exact_action() -> None:
    actor = principal()
    action = Action("refund", {"order_id": "A-17"}, irreversible=True)
    approval = Approval("reviewer-9", action_digest(action, actor), POLICY_VERSION)
    state, audit = world(), AuditLog()
    decision = EnforcementPoint(PolicyEngine(), audit).execute(action, actor, state, approval)
    assert decision.effect == "allow"
    assert state["refunds"] == ["A-17"]


def test_substituted_action_is_rejected() -> None:
    actor = principal()
    approved = Action("refund", {"order_id": "A-17"}, irreversible=True)
    substituted = Action("refund", {"order_id": "A-99"}, irreversible=True)
    token = Approval("reviewer-9", action_digest(approved, actor), POLICY_VERSION)
    state = world()
    decision = EnforcementPoint(PolicyEngine(), AuditLog()).execute(substituted, actor, state, token)
    assert decision.effect == "deny"
    assert state["refunds"] == []


def test_egress_is_allowlisted() -> None:
    action = Action("render_url", {"url": "https://evil.example/collect"})
    state = world()
    decision = EnforcementPoint(PolicyEngine(), AuditLog()).execute(action, principal(), state)
    assert decision.effect == "deny"
    assert state["fetched"] == []


def test_audit_chain_detects_tampering() -> None:
    audit = AuditLog()
    gate = EnforcementPoint(PolicyEngine(), audit)
    gate.execute(Action("render_url", {"url": "https://evil.example/x"}), principal(), world())
    assert audit.verify()
    audit.entries[0]["decision"]["reason"] = "rewritten"
    assert not audit.verify()
