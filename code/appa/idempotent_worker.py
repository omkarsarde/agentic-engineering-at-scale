"""Duplicate-delivery and crash-window lab for Appendix A."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class InjectedCrash(RuntimeError):
    """Raised once after an external effect and before local commit."""


@dataclass(frozen=True)
class Task:
    """Canonical notification intent carried by an at-least-once queue."""

    payment_id: str
    recipient: str

    @property
    def key(self) -> str:
        """Return an idempotency key derived from intent, not delivery attempt."""

        payload = json.dumps(
            {"payment_id": self.payment_id, "recipient": self.recipient},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class CountingProvider:
    """Count calls and effects, optionally honoring caller idempotency keys."""

    def __init__(self, deduplicate: bool = False) -> None:
        self.deduplicate = deduplicate
        self.calls = 0
        self.effects = 0
        self.seen: set[str] = set()

    def send(self, task: Task) -> None:
        """Record one provider call and possibly one externally visible effect."""

        self.calls += 1
        if self.deduplicate and task.key in self.seen:
            return
        self.seen.add(task.key)
        self.effects += 1


class Ledger:
    """Persist pending and committed intent identities in SQLite."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )
        self.connection.commit()

    def status(self, key: str) -> str | None:
        """Return current status for an intent key."""

        row = self.connection.execute("SELECT status FROM effects WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def reserve(self, key: str) -> bool:
        """Return true when work may run; committed work is ineligible."""

        if self.status(key) == "committed":
            return False
        self.connection.execute(
            "INSERT OR IGNORE INTO effects(key, status) VALUES (?, 'pending')", (key,)
        )
        self.connection.commit()
        return True

    def commit(self, key: str) -> None:
        """Mark a previously reserved intent committed."""

        self.connection.execute("UPDATE effects SET status = 'committed' WHERE key = ?", (key,))
        self.connection.commit()

    def histogram(self) -> dict[str, int]:
        """Return count by ledger status."""

        rows = self.connection.execute(
            "SELECT status, COUNT(*) FROM effects GROUP BY status ORDER BY status"
        ).fetchall()
        return {str(status): int(count) for status, count in rows}


def fixture_deliveries(task_count: int = 20) -> list[Task]:
    """Return unique tasks plus one duplicate after every fifth task."""

    unique = [Task(f"payment-{index:02d}", f"user-{index:02d}@example.test") for index in range(task_count)]
    deliveries: list[Task] = []
    for index, task in enumerate(unique, start=1):
        deliveries.append(task)
        if index % 5 == 0:
            deliveries.append(task)
    return deliveries


def run_naive(deliveries: Iterable[Task], provider: CountingProvider) -> None:
    """Execute every delivery with no intent ledger."""

    for task in deliveries:
        provider.send(task)


class IdempotentWorker:
    """Collapse duplicate intents and expose the effect-before-commit window."""

    def __init__(self, ledger: Ledger, provider: CountingProvider) -> None:
        self.ledger = ledger
        self.provider = provider
        self.crashed: set[str] = set()

    def process(self, task: Task, crash_after_effect_for: str | None = None) -> str:
        """Process one delivery, optionally crashing once in the ambiguous window."""

        if not self.ledger.reserve(task.key):
            return "duplicate:committed"
        self.provider.send(task)
        if crash_after_effect_for == task.payment_id and task.key not in self.crashed:
            self.crashed.add(task.key)
            raise InjectedCrash(task.payment_id)
        self.ledger.commit(task.key)
        return "committed"

    def drain(self, deliveries: Iterable[Task], crash_after_effect_for: str | None = None) -> int:
        """Drain deliveries and return injected crash count."""

        crashes = 0
        retry: list[Task] = []
        for task in deliveries:
            try:
                self.process(task, crash_after_effect_for)
            except InjectedCrash:
                crashes += 1
                retry.append(task)
        for task in retry:
            self.process(task, crash_after_effect_for)
        return crashes


def build_report() -> dict[str, object]:
    """Run duplicate, local-ledger, ambiguous-window, and provider-key drills."""

    deliveries = fixture_deliveries()
    naive = CountingProvider()
    run_naive(deliveries, naive)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        local_provider = CountingProvider()
        local_ledger = Ledger(root / "local.sqlite")
        IdempotentWorker(local_ledger, local_provider).drain(deliveries)

        ambiguous_provider = CountingProvider()
        ambiguous_ledger = Ledger(root / "ambiguous.sqlite")
        ambiguous_crashes = IdempotentWorker(ambiguous_ledger, ambiguous_provider).drain(
            deliveries, "payment-07"
        )

        dedupe_provider = CountingProvider(deduplicate=True)
        dedupe_ledger = Ledger(root / "dedupe.sqlite")
        dedupe_crashes = IdempotentWorker(dedupe_ledger, dedupe_provider).drain(
            deliveries, "payment-07"
        )

        return {
            "unique_tasks": 20,
            "deliveries": len(deliveries),
            "naive": {"calls": naive.calls, "effects": naive.effects},
            "local_ledger": {
                "calls": local_provider.calls,
                "effects": local_provider.effects,
                "states": local_ledger.histogram(),
            },
            "ambiguous_window": {
                "crashes": ambiguous_crashes,
                "calls": ambiguous_provider.calls,
                "effects": ambiguous_provider.effects,
                "states": ambiguous_ledger.histogram(),
            },
            "provider_key": {
                "crashes": dedupe_crashes,
                "calls": dedupe_provider.calls,
                "effects": dedupe_provider.effects,
                "states": dedupe_ledger.histogram(),
            },
        }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
