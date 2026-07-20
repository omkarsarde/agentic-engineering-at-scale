"""Integrated public API and grounded-answer boundary for Chapter 14.

Corpus types, ranked metrics, and retrieval stages live in focused sibling
modules. This facade keeps one end-to-end import surface for the chapter build.
"""

from __future__ import annotations

import re
from typing import Sequence

from hybrid_retrieval import (
    BM25,
    DenseIndex,
    dense_terms,
    embed,
    pair_score,
    query_variants,
    rerank,
    retrieve,
    rrf_merge,
)
from rag_types import (
    CANON,
    CODE_RE,
    STOP,
    TOKEN_RE,
    Chunk,
    Document,
    Hit,
    chunk_documents,
    load_documents,
    load_jsonl,
    terms,
)
from retrieval_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def assemble_context(hits: Sequence[Hit], lookup: dict[str, Chunk], budget: int = 220) -> str:
    """Render versioned, citable source blocks without splitting a chunk."""
    blocks, used = [], 0
    for hit in hits:
        chunk = lookup[hit.chunk_id]
        block = f'<source id="{chunk.chunk_id}" version="{chunk.version}" trust="retrieved-data">\n{chunk.text}\n</source>'
        size = len(block.split())
        if used + size > budget:
            continue
        blocks.append(block)
        used += size
    return "\n".join(blocks)


def answer_or_abstain(query: str, hits: Sequence[Hit], lookup: dict[str, Chunk], threshold: float = 3.0) -> dict:
    """Extract one supported sentence or take the insufficient-evidence path."""
    if not hits or hits[0].score < threshold:
        return {"answer": "INSUFFICIENT_EVIDENCE", "citations": [], "abstained": True}
    candidates: list[tuple[float, str, str]] = []
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?])\s+", lookup[hit.chunk_id].text):
            candidate = Chunk("sentence", hit.source_id, "", "", 0, sentence)
            candidates.append((pair_score(query, candidate), sentence, hit.chunk_id))
    _, sentence, chunk_id = max(candidates, key=lambda row: (row[0], row[1]))
    return {"answer": f"{sentence} [{chunk_id}]", "citations": [chunk_id], "abstained": False}


def citation_support(answer: str, lookup: dict[str, Chunk]) -> float:
    """Score whether cited chunks literally support the answer's uncited claim."""
    cited = re.findall(r"\[([^\]]+)\]", answer)
    if not cited or any(chunk_id not in lookup for chunk_id in cited):
        return 0.0
    claim = re.sub(r"\[[^\]]+\]", "", answer).strip().lower()
    return float(any(claim.rstrip(".") in lookup[chunk_id].text.lower().rstrip(".") for chunk_id in cited))
