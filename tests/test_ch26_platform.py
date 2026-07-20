"""Focused invariants for the Chapter 26 mini platform."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch26"
sys.path.insert(0, str(CODE))

from cost_curves import cached_cost, detection_probability, uncached_cost  # noqa: E402
from mini_platform import (  # noqa: E402
    BudgetError,
    Deployment,
    EffectLedger,
    Gateway,
    IdempotentProvider,
    InjectedCrash,
    TokenBucket,
    Usage,
    bundle_digest,
    canary_gate,
    execute_once,
    fallback_allowed,
    run_demo,
)


def gateway() -> Gateway:
    return Gateway(
        {"key-a": ("tenant-a", frozenset({"chat"}))},
        {"tenant-a": 0.01},
        [Deployment("cheap", frozenset({"chat"}), 1.0)],
    )


def test_identity_and_cost_are_derived_from_virtual_key() -> None:
    gate = gateway()
    context = gate.authenticate("key-a", "request-1", 10.0)
    assert context.tenant == "tenant-a"
    assert gate.route(context, "chat").name == "cheap"
    assert gate.settle(context, Usage(1_000, 100, 1.0, 2.0)) == 0.0012


def test_budget_blocks_whole_journey_before_call() -> None:
    gate = gateway()
    context = gate.authenticate("key-a", "request-1", 10.0)
    try:
        gate.admit(context, 0.02)
    except BudgetError:
        pass
    else:
        raise AssertionError("journey above tenant budget was admitted")


def test_policy_refusal_never_provider_shops() -> None:
    assert not fallback_allowed("policy_refusal", 5.0)
    assert fallback_allowed("overloaded", 5.0)
    assert not fallback_allowed("overloaded", 0.0)


def test_three_crashes_still_create_one_effect() -> None:
    report = run_demo()
    assert report["injected_crashes"] == 3
    assert report["provider_effects"] == 1
    assert report["ledger_state"] == "RECORDED"


def test_same_effect_key_cannot_change_payload() -> None:
    ledger, provider = EffectLedger(), IdempotentProvider()
    execute_once(ledger, provider, "refund:A", {"amount": 10})
    try:
        execute_once(ledger, provider, "refund:A", {"amount": 20})
    except ValueError:
        pass
    else:
        raise AssertionError("same effect identity accepted a substituted payload")


def test_token_bucket_applies_backpressure() -> None:
    clock = [0.0]
    bucket = TokenBucket(100, 10, lambda: clock[0])
    assert bucket.allow(80)
    assert not bucket.allow(30)
    clock[0] = 1.0
    assert bucket.allow(30)


def test_release_bundle_hash_covers_every_surface() -> None:
    surface = {"model": "m1", "prompt": "p1", "corpus": "c1"}
    first = bundle_digest(surface)
    assert first == bundle_digest(dict(reversed(list(surface.items()))))
    assert first != bundle_digest({**surface, "corpus": "c2"})


def test_cache_and_canary_math() -> None:
    assert cached_cost(1, 1.0, 1.25, 0.10) < uncached_cost(2, 1.0)
    assert 0.71 < detection_probability(25, 0.05) < 0.73
    assert canary_gate(98, 100, 0.90).effect == "PROMOTE"
    assert canary_gate(24, 25, 0.90).effect == "HOLD"
    assert canary_gate(100, 100, 0.90, critical_failures=1).effect == "HOLD"
