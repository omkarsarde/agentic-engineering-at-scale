"""Workspace and resumable audit support for the Chapter 17 harness."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Workspace:
    """Resolve paths inside one workspace; this is not an OS sandbox."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        """Return a contained path or reject traversal and absolute escapes."""
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"path escapes workspace: {relative}")
        return candidate


class Journal:
    """Checkpoint thread state and deduplicate harness audit events."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(event_id TEXT PRIMARY KEY, thread_id TEXT, kind TEXT, payload TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS threads "
            "(thread_id TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )

    def record(self, event_id: str, thread_id: str, kind: str, payload: Any) -> bool:
        """Append one idempotent audit event and report whether it was new."""
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?)",
            (event_id, thread_id, kind, json.dumps(payload, sort_keys=True)),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def checkpoint(self, thread_id: str, state: Any) -> None:
        """Replace harness state; this does not make external effects durable."""
        self.db.execute(
            "INSERT INTO threads VALUES (?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET state=excluded.state",
            (thread_id, json.dumps(state, sort_keys=True)),
        )
        self.db.commit()

    def load(self, thread_id: str) -> Any:
        """Load the latest harness checkpoint."""
        row = self.db.execute(
            "SELECT state FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def event_count(self, thread_id: str) -> int:
        """Count durable audit rows for a thread."""
        return self.db.execute(
            "SELECT COUNT(*) FROM events WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
