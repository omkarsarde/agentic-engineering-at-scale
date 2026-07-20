"""Permission-aware graph retrieval with bounded, typed outcomes."""

from __future__ import annotations

from evidence_index import Document, EvidenceIndex, Query, fact_supported
from retrieval_contract import EvidenceEdge, RetrievalResult, Stop


def _verified_edge(
    documents: tuple[Document, ...], subject: str, relation: str
) -> tuple[EvidenceEdge | None, tuple[str, ...], bool]:
    """Resolve one supported edge, exposing rejected and conflicting candidates."""
    rejected: list[str] = []
    supported: list[EvidenceEdge] = []
    for document in documents:
        for fact in document.facts:
            if fact.subject == subject and fact.relation == relation:
                if fact_supported(document, fact):
                    supported.append(
                        EvidenceEdge(
                            fact.subject,
                            fact.relation,
                            fact.object,
                            document.id,
                            document.source_uri,
                            fact.evidence,
                        )
                    )
                else:
                    rejected.append(document.id)
    objects = {edge.object for edge in supported}
    if len(objects) > 1:
        return None, tuple(dict.fromkeys(rejected)), True
    return (
        supported[0] if supported else None,
        tuple(dict.fromkeys(rejected)),
        False,
    )


def one_shot(index: EvidenceIndex, query: Query, k: int = 1) -> RetrievalResult:
    """Retrieve once, then follow only edges present in that fixed context."""
    documents = index.lexical(query.text, query, k=k)
    current = query.start
    evidence: list[EvidenceEdge] = []
    rejected: list[str] = []
    trace = [f"lexical:{','.join(doc.id for doc in documents) or '-'}"]
    for relation in query.relations:
        edge, bad, conflict = _verified_edge(documents, current, relation)
        rejected.extend(bad)
        if conflict:
            return RetrievalResult(
                Stop.CONFLICT,
                None,
                tuple(evidence),
                1,
                len(documents),
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        if edge is None:
            return RetrievalResult(
                Stop.ABSTAINED,
                None,
                tuple(evidence),
                1,
                len(documents),
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        evidence.append(edge)
        current = edge.object
        trace.append(f"edge:{relation}->{current}")
    return RetrievalResult(
        Stop.ANSWERED,
        current,
        tuple(evidence),
        1,
        len(documents),
        tuple(dict.fromkeys(rejected)),
        tuple(trace),
    )


def agentic(
    index: EvidenceIndex,
    query: Query,
    max_search_calls: int = 4,
    max_candidates_per_call: int = 8,
) -> RetrievalResult:
    """Plan one edge at a time, verify evidence, and stop within a hard budget."""
    if max_search_calls < 0:
        raise ValueError("max_search_calls cannot be negative")
    if max_candidates_per_call < 1:
        raise ValueError("max_candidates_per_call must be positive")
    current = query.start
    evidence: list[EvidenceEdge] = []
    rejected: list[str] = []
    trace: list[str] = [f"route:{'direct' if len(query.relations) == 1 else 'multi-hop'}"]
    calls = 0
    candidate_documents = 0
    for relation in query.relations:
        if calls >= max_search_calls:
            return RetrievalResult(
                Stop.BUDGET_EXHAUSTED,
                None,
                tuple(evidence),
                calls,
                candidate_documents,
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        batch = index.fact_candidates(
            current, relation, query, limit=max_candidates_per_call
        )
        calls += 1
        candidates = batch.documents
        candidate_documents += len(candidates)
        rejected.extend(batch.rejected_ids)
        trace.append(
            f"search:{current}/{relation}:{','.join(d.id for d in candidates) or '-'}"
        )
        if batch.truncated:
            trace.append("stop:candidate_limit")
            return RetrievalResult(
                Stop.CANDIDATE_LIMIT,
                None,
                tuple(evidence),
                calls,
                candidate_documents,
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        edge, bad, conflict = _verified_edge(candidates, current, relation)
        rejected.extend(bad)
        if conflict:
            return RetrievalResult(
                Stop.CONFLICT,
                None,
                tuple(evidence),
                calls,
                candidate_documents,
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        if edge is None:
            return RetrievalResult(
                Stop.ABSTAINED,
                None,
                tuple(evidence),
                calls,
                candidate_documents,
                tuple(dict.fromkeys(rejected)),
                tuple(trace),
            )
        current = edge.object
        evidence.append(edge)
        trace.append(f"verify:{edge.document_id}->{current}")
    return RetrievalResult(
        Stop.ANSWERED,
        current,
        tuple(evidence),
        calls,
        candidate_documents,
        tuple(dict.fromkeys(rejected)),
        tuple(trace),
    )


def correct(result: RetrievalResult, query: Query, index: EvidenceIndex) -> bool:
    """Re-verify complete evidence chains and required abstentions for scoring."""
    if query.expected is None:
        return result.stop == Stop.ABSTAINED and result.answer is None
    if result.stop != Stop.ANSWERED or len(result.evidence) != len(query.relations):
        return False
    documents = {document.id: document for document in index.documents}
    current = query.start
    for relation, edge in zip(query.relations, result.evidence):
        if (
            edge.subject != current
            or edge.relation != relation
            or not edge.document_id
            or not edge.source_uri
            or not edge.support
        ):
            return False
        source = documents.get(edge.document_id)
        if source is None or source.source_uri != edge.source_uri:
            return False
        matching = (
            fact
            for fact in source.facts
            if (fact.subject, fact.relation, fact.object, fact.evidence)
            == (edge.subject, edge.relation, edge.object, edge.support)
        )
        if not any(fact_supported(source, fact) for fact in matching):
            return False
        current = edge.object
    return result.answer == query.expected == current
