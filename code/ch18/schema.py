"""Typed records for the Chapter 18 memory artifact."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    """Semantic content, events, procedures, and short-lived working state."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class Source(StrEnum):
    """Provenance class used by the write policy."""

    USER = "user"
    VERIFIED_TOOL = "verified_tool"
    MODEL_INFERENCE = "model_inference"
    RETRIEVED_DOCUMENT = "retrieved_document"


class Status(StrEnum):
    """Lifecycle state of one immutable record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


@dataclass(frozen=True)
class Scope:
    """Ownership dimensions checked before retrieval."""

    tenant_id: str
    user_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A proposed memory write; policy may reject or transform it."""

    key: str
    value: str
    kind: Kind
    scope: Scope
    source: Source
    evidence_id: str
    event_time: int
    confidence: float = 1.0


@dataclass(frozen=True)
class Record:
    """An immutable temporal record with explicit provenance and status."""

    record_id: str
    key: str
    value: str
    kind: Kind
    scope: Scope
    source: Source
    evidence_id: str
    confidence: float
    valid_from: int
    valid_to: int | None = None
    status: Status = Status.ACTIVE


@dataclass(frozen=True)
class DeletionManifest:
    """Evidence that one subject was removed from each local projection."""

    subject: str
    record_ids: tuple[str, ...]
    targets: dict[str, str]
