"""Integrated public API and deterministic build for the Chapter 26 platform.

The implementation is divided by operational boundary: ``gateway_control``
owns identity and admission, ``effect_runtime`` owns effects and backpressure,
and ``release_gate`` owns the canary decision. This facade keeps the chapter's
single import surface and executable build.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from effect_runtime import (
    EffectLedger,
    EffectRecord,
    EffectState,
    IdempotentProvider,
    InjectedCrash,
    TokenBucket,
    bundle_digest,
    execute_once,
)
from gateway_control import (
    RETRYABLE_ERRORS,
    TERMINAL_ERRORS,
    BudgetError,
    Deployment,
    Gateway,
    RequestContext,
    RetryBudget,
    Usage,
    fallback_allowed,
)
from release_gate import CanaryDecision, canary_gate, wilson_lower


__all__ = [
    "BudgetError",
    "CanaryDecision",
    "Deployment",
    "EffectLedger",
    "EffectRecord",
    "EffectState",
    "Gateway",
    "IdempotentProvider",
    "InjectedCrash",
    "RETRYABLE_ERRORS",
    "RequestContext",
    "RetryBudget",
    "TERMINAL_ERRORS",
    "TokenBucket",
    "Usage",
    "bundle_digest",
    "canary_gate",
    "execute_once",
    "fallback_allowed",
    "run_demo",
    "wilson_lower",
]


def run_demo() -> dict[str, Any]:
    """Exercise identity, cost attribution, crash recovery, and release gating."""
    gateway = Gateway(
        {"vk-acme": ("acme", frozenset({"chat"}))},
        {"acme": 0.05},
        [
            Deployment("small", frozenset({"chat"}), 1.0),
            Deployment("large", frozenset({"chat", "vision"}), 5.0),
        ],
    )
    context = gateway.authenticate("vk-acme", "req-7", deadline_s=30.0)
    usage = Usage(20_000, 2_000, 0.50, 1.50)
    gateway.admit(context, usage.cost)
    route = gateway.route(context, "chat")
    spend = gateway.settle(context, usage)

    ledger, provider = EffectLedger(), IdempotentProvider()
    payload = {"order_id": "A-17", "amount": 25}
    crashes = 0
    for _ in range(3):
        try:
            execute_once(ledger, provider, "refund:A-17:v1", payload, True)
        except InjectedCrash:
            crashes += 1
    receipt = execute_once(ledger, provider, "refund:A-17:v1", payload)

    decision = canary_gate(98, 100, minimum_rate=0.90)
    return {
        "tenant": context.tenant,
        "route": route.name,
        "journey_cost": round(spend, 6),
        "injected_crashes": crashes,
        "provider_effects": provider.effect_count,
        "receipt": receipt,
        "ledger_state": ledger.records["refund:A-17:v1"].state.value,
        "canary": asdict(decision),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
