"""Typed outcomes and edge-level evidence for Chapter 15."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stop(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANDIDATE_LIMIT = "candidate_limit"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class EvidenceEdge:
    """One resolved relation and the exact source span supporting it."""

    subject: str
    relation: str
    object: str
    document_id: str
    source_uri: str
    support: str


@dataclass(frozen=True)
class RetrievalResult:
    """A typed answer together with cost, evidence, and rejection trace."""

    stop: Stop
    answer: str | None
    evidence: tuple[EvidenceEdge, ...]
    search_calls: int
    candidate_documents: int
    rejected_ids: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()

    @property
    def citations(self) -> tuple[str, ...]:
        """Preserve one citation per edge; repeated document ids are meaningful."""
        return tuple(edge.document_id for edge in self.evidence)
