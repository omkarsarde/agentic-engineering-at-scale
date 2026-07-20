"""Crash-safe effects and load controls for the Chapter 26 mini platform."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class EffectState(str, Enum):
    INTENDED = "INTENDED"
    RESERVED = "RESERVED"
    EXECUTED = "EXECUTED"
    RECORDED = "RECORDED"
    FAILED = "FAILED"


@dataclass
class EffectRecord:
    key: str
    payload_digest: str
    state: EffectState
    receipt: str | None = None


class EffectLedger:
    """Keep stable effect identity across crash and workflow replay."""

    def __init__(self) -> None:
        self.records: dict[str, EffectRecord] = {}

    @staticmethod
    def digest(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def reserve(self, key: str, payload: dict[str, Any]) -> EffectRecord:
        digest = self.digest(payload)
        record = self.records.get(key)
        if record and record.payload_digest != digest:
            raise ValueError("idempotency key reused for different payload")
        if record is None:
            record = EffectRecord(key, digest, EffectState.INTENDED)
            self.records[key] = record
        if record.state == EffectState.INTENDED:
            record.state = EffectState.RESERVED
        return record


class InjectedCrash(RuntimeError):
    """Simulate worker death after a provider accepted an effect."""


class IdempotentProvider:
    """A provider that returns the same receipt for a repeated stable key."""

    def __init__(self) -> None:
        self.receipts: dict[str, str] = {}
        self.effect_count = 0

    def apply(self, key: str, payload: dict[str, Any]) -> str:
        if key not in self.receipts:
            self.effect_count += 1
            self.receipts[key] = f"receipt-{self.effect_count}"
        return self.receipts[key]


def execute_once(
    ledger: EffectLedger,
    provider: IdempotentProvider,
    key: str,
    payload: dict[str, Any],
    crash_after_provider: bool = False,
) -> str:
    """Execute or recover one effect using a stable application-owned key."""
    record = ledger.reserve(key, payload)
    if record.state == EffectState.RECORDED:
        assert record.receipt is not None
        return record.receipt

    try:
        receipt = provider.apply(key, payload)
        if crash_after_provider:
            raise InjectedCrash("worker died before recording provider receipt")
        record.state = EffectState.EXECUTED
        record.receipt = receipt
        record.state = EffectState.RECORDED
        return receipt
    except InjectedCrash:
        raise
    except Exception:
        record.state = EffectState.FAILED
        raise


class TokenBucket:
    """Enforce a token-rate envelope with an injected monotonic clock."""

    def __init__(
        self, capacity: float, refill_per_s: float, now: Callable[[], float]
    ) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.now = now
        self.tokens = capacity
        self.updated_at = now()

    def allow(self, requested_tokens: float) -> bool:
        current = self.now()
        elapsed = max(0.0, current - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.updated_at = current
        if requested_tokens > self.tokens:
            return False
        self.tokens -= requested_tokens
        return True


def bundle_digest(surface: dict[str, str]) -> str:
    """Address a complete release bundle by canonical content hash."""
    encoded = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
