"""A scoped, temporal memory policy for Chapter 18."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Iterable

from schema import Candidate, DeletionManifest, Kind, Record, Scope, Source, Status


def _tokens(text: str) -> set[str]:
    """Normalize words for the deterministic retrieval probe."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _record_id(candidate: Candidate) -> str:
    """Create stable identity from provenance and owned scope."""
    raw = "|".join(
        (
            candidate.scope.tenant_id,
            candidate.scope.user_id or "",
            candidate.key,
            candidate.evidence_id,
            str(candidate.event_time),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _visible(record: Record, scope: Scope) -> bool:
    """Apply tenant and optional subject/agent/task scope before scoring."""
    if record.scope.tenant_id != scope.tenant_id:
        return False
    for field in ("user_id", "agent_id", "task_id"):
        owned = getattr(record.scope, field)
        if owned is not None and owned != getattr(scope, field):
            return False
    return True


class MemoryPolicy:
    """Validate writes and reject instruction-like untrusted memories."""

    forbidden_phrases = ("ignore previous", "system prompt", "wire money")

    def validate(self, candidate: Candidate) -> tuple[bool, str]:
        """Return a policy decision for one candidate write."""
        if not candidate.evidence_id:
            return False, "missing provenance"
        if not 0 <= candidate.confidence <= 1:
            return False, "confidence outside [0, 1]"
        lowered = candidate.value.casefold()
        if candidate.source is Source.RETRIEVED_DOCUMENT:
            return False, "retrieved text cannot write durable memory directly"
        if any(phrase in lowered for phrase in self.forbidden_phrases):
            return False, "instruction-like memory requires review"
        if candidate.kind is Kind.PROCEDURAL and candidate.source is Source.MODEL_INFERENCE:
            return False, "model inference cannot self-author a procedure"
        return True, "accepted"


class MemoryStore:
    """Own records plus deterministic search and cache projections."""

    def __init__(self, policy: MemoryPolicy | None = None) -> None:
        self.policy = policy or MemoryPolicy()
        self.records: list[Record] = []
        self.search_index: dict[str, set[str]] = {}
        self.cache: dict[tuple[str, str], str] = {}

    def write(self, candidate: Candidate) -> tuple[bool, str | Record]:
        """Validate, supersede a conflicting fact, and index a new record."""
        allowed, reason = self.policy.validate(candidate)
        if not allowed:
            return False, reason

        if candidate.kind is Kind.SEMANTIC:
            for index, old in enumerate(self.records):
                same_fact = old.key == candidate.key and old.scope == candidate.scope
                if same_fact and old.status is Status.ACTIVE:
                    self.records[index] = replace(
                        old, status=Status.SUPERSEDED, valid_to=candidate.event_time
                    )

        record = Record(
            _record_id(candidate),
            candidate.key,
            candidate.value,
            candidate.kind,
            candidate.scope,
            candidate.source,
            candidate.evidence_id,
            candidate.confidence,
            candidate.event_time,
        )
        self.records.append(record)
        for token in _tokens(f"{record.key} {record.value}"):
            self.search_index.setdefault(token, set()).add(record.record_id)
        self.cache.clear()
        return True, record

    def retrieve(
        self,
        query: str,
        scope: Scope,
        as_of: int | None = None,
        threshold: float = 1.25,
    ) -> Record | None:
        """Return the best visible temporal record, or abstain below threshold."""
        cache_key = (scope.user_id or "", query.casefold())
        query_tokens = _tokens(query)
        candidates: list[tuple[float, Record]] = []
        for record in self.records:
            current = record.status is Status.ACTIVE
            temporal = as_of is not None and record.valid_from <= as_of and (
                record.valid_to is None or as_of < record.valid_to
            )
            if not _visible(record, scope) or not (current if as_of is None else temporal):
                continue
            overlap = len(query_tokens & _tokens(f"{record.key} {record.value}"))
            score = overlap + record.confidence
            candidates.append((score, record))
        if not candidates:
            return None
        score, record = max(candidates, key=lambda item: (item[0], item[1].valid_from))
        if score < threshold:
            return None
        self.cache[cache_key] = record.record_id
        return record

    def consolidate(self, records: Iterable[Record], candidate: Candidate) -> bool:
        """Write a derived fact only when two verified episodes support it."""
        evidence = [
            record
            for record in records
            if record.kind is Kind.EPISODIC
            and record.source is Source.VERIFIED_TOOL
            and record.status is Status.ACTIVE
        ]
        if len(evidence) < 2 or candidate.kind is not Kind.SEMANTIC:
            return False
        return self.write(candidate)[0]

    def delete_user(self, tenant_id: str, user_id: str) -> DeletionManifest:
        """Delete one subject from records, search projection, and cache."""
        deleted: list[str] = []
        for index, record in enumerate(self.records):
            owned = record.scope.tenant_id == tenant_id and record.scope.user_id == user_id
            if owned and record.status is not Status.DELETED:
                deleted.append(record.record_id)
                self.records[index] = replace(record, status=Status.DELETED)
        for ids in self.search_index.values():
            ids.difference_update(deleted)
        self.cache = {key: value for key, value in self.cache.items() if value not in deleted}
        targets = {"primary_store": "deleted", "search_index": "deleted", "cache": "deleted"}
        return DeletionManifest(user_id, tuple(deleted), targets)
