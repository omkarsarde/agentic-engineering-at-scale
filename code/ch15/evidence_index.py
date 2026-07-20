"""Typed evidence records and bounded indexes for Chapter 15."""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass


TOKEN = re.compile(r"[a-z0-9-]+")
NEGATION = re.compile(r"\b(?:no|not|never|without)\b")
RELATION_CUES = {
    "owner": ("owned by",),
    "depends_on": ("depends on",),
    "telemetry_store": ("stores its telemetry in",),
    "region": ("runs in region",),
    "release_gate": ("release gate for",),
    "annual_budget": ("annual budget",),
}


@dataclass(frozen=True)
class Fact:
    """A provenance-bearing edge extracted during ingestion."""

    subject: str
    relation: str
    object: str
    evidence: str = ""


@dataclass(frozen=True)
class Document:
    """A retrievable record with access and integrity metadata."""

    id: str
    tenant: str
    groups: frozenset[str]
    source_uri: str
    content: str
    facts: tuple[Fact, ...]
    integrity: str = "unverified"


@dataclass(frozen=True)
class CandidateBatch:
    """A bounded candidate page that makes truncation explicit."""

    documents: tuple[Document, ...]
    rejected_ids: tuple[str, ...]
    truncated: bool


@dataclass(frozen=True)
class Query:
    """A compiled retrieval request used by the deterministic fixture."""

    id: str
    text: str
    tenant: str
    groups: frozenset[str]
    start: str
    relations: tuple[str, ...]
    expected: str | None

    def __post_init__(self) -> None:
        if not self.start.strip() or not self.relations:
            raise ValueError("a retrieval query needs a start entity and at least one relation")


def _tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def fact_supported(document: Document, fact: Fact) -> bool:
    """Verify one fixture relation with a trusted span and an allowlisted grammar."""
    support = _normalized(fact.evidence)
    if document.integrity != "verified" or not support:
        return False
    cues = RELATION_CUES.get(fact.relation, ())
    negated = bool(NEGATION.search(support))
    return all(
        (
            support in _normalized(document.content),
            _normalized(fact.subject) in support,
            _normalized(fact.object) in support,
            bool(cues) and any(cue in support for cue in cues),
            not negated,
        )
    )


def _candidate_key(document: Document) -> tuple[bool, str]:
    return document.integrity != "verified", document.id


class EvidenceIndex:
    """Small lexical and fact index; production adapters keep this contract."""

    def __init__(self, documents: tuple[Document, ...]):
        if len({doc.id for doc in documents}) != len(documents):
            raise ValueError("document ids must be unique")
        self.documents = documents
        partitions: dict[tuple[str, str, str, str], list[Document]] = {}
        for document in documents:
            edge_keys = {(fact.subject, fact.relation) for fact in document.facts}
            for subject, relation in edge_keys:
                for group in document.groups:
                    key = (subject, relation, document.tenant, group)
                    partitions.setdefault(key, []).append(document)
        self._fact_partitions = {
            key: tuple(sorted(value, key=_candidate_key))
            for key, value in partitions.items()
        }

    @staticmethod
    def allowed(document: Document, query: Query) -> bool:
        """Apply tenant and group predicates before scoring or graph expansion."""
        same_scope = document.tenant in {query.tenant, "public"}
        group_match = "all" in document.groups or bool(document.groups & query.groups)
        return same_scope and group_match

    def lexical(self, text: str, query: Query, k: int = 1) -> tuple[Document, ...]:
        """Return admitted, authorized documents by deterministic token overlap."""
        if k < 1:
            raise ValueError("k must be positive")
        terms = _tokens(text)
        candidates = [
            doc
            for doc in self.documents
            if self.allowed(doc, query)
            and doc.integrity == "verified"
        ]
        ranked = sorted(
            candidates,
            key=lambda doc: (-len(terms & _tokens(doc.content)), doc.id),
        )
        return tuple(doc for doc in ranked[:k] if terms & _tokens(doc.content))

    def fact_candidates(
        self, subject: str, relation: str, query: Query, limit: int = 8
    ) -> CandidateBatch:
        """Expand one edge from ACL partitions with a typed volume bound."""
        if limit < 1:
            raise ValueError("candidate limit must be positive")
        tenants = (query.tenant,) if query.tenant == "public" else (query.tenant, "public")
        groups = tuple(sorted(set(query.groups) | {"all"}))
        streams = [
            self._fact_partitions.get((subject, relation, tenant, group), ())
            for tenant in tenants
            for group in groups
        ]
        selected: list[Document] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for document in heapq.merge(*streams, key=_candidate_key):
            if document.id in seen:
                continue
            seen.add(document.id)
            if document.integrity != "verified":
                rejected.append(document.id)
                if len(rejected) == limit:
                    break
                continue
            selected.append(document)
            if len(selected) > limit:
                break
        return CandidateBatch(
            tuple(selected[:limit]), tuple(rejected), len(selected) > limit
        )
