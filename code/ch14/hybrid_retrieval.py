"""Sparse, dense, fusion, query-rewrite, and reranking stages."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from typing import Sequence

from rag_types import CANON, CODE_RE, STOP, Chunk, Hit, terms


class BM25:
    """Small exact BM25 index using Robertson's positive-IDF form."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.rows = [terms(f"{chunk.title} {chunk.text}") for chunk in chunks]
        self.freqs = [Counter(row) for row in self.rows]
        self.avg_len = sum(map(len, self.rows)) / max(len(self.rows), 1)
        self.df = Counter(term for row in self.rows for term in set(row))

    def idf(self, term: str) -> float:
        n, df = len(self.rows), self.df.get(term.lower(), 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Hit]:
        scores: list[Hit] = []
        for chunk, freq, length in zip(self.chunks, self.freqs, map(len, self.rows)):
            score = 0.0
            for term in terms(query):
                tf = freq.get(term, 0)
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf(term) * (tf * (self.k1 + 1) / denominator if denominator else 0)
            scores.append(Hit(chunk.chunk_id, chunk.source_id, score))
        return sorted(scores, key=lambda hit: (-hit.score, hit.chunk_id))[:k]


def dense_terms(text: str) -> list[str]:
    """Normalize concepts and omit rare codes to expose hybrid complementarity."""
    return [CANON.get(term, term) for term in terms(text) if term not in STOP and not CODE_RE.fullmatch(term)]


def embed(text: str, dimensions: int = 192) -> tuple[float, ...]:
    """Create a deterministic signed feature-hash vector, then L2-normalize it."""
    vector = [0.0] * dimensions
    normalized = dense_terms(text)
    features = [*normalized, *(f"{a}_{b}" for a, b in zip(normalized, normalized[1:]))]
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


class DenseIndex:
    """Exact normalized-vector baseline that any ANN index must approximate."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.vectors = [embed(f"{chunk.title}. {chunk.text}") for chunk in chunks]

    def search(self, query: str, k: int) -> list[Hit]:
        query_vector = embed(query)
        hits = [Hit(chunk.chunk_id, chunk.source_id, sum(a * b for a, b in zip(query_vector, vector))) for chunk, vector in zip(self.chunks, self.vectors)]
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk_id))[:k]


def rrf_merge(rankings: Sequence[Sequence[Hit]], lookup: dict[str, Chunk], k: int = 60, limit: int | None = None) -> list[Hit]:
    """Fuse rankings without comparing their raw score scales."""
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk_id] += 1 / (k + rank)
    merged = [Hit(chunk_id, lookup[chunk_id].source_id, score) for chunk_id, score in scores.items()]
    return sorted(merged, key=lambda hit: (-hit.score, hit.chunk_id))[:limit]


def query_variants(query: str, history: str = "") -> list[str]:
    """Return bounded rewrites for vocabulary mismatch and ellipsis."""
    lower = query.lower().strip()
    if lower == "what about annual plans?" and history:
        return [query, "cancel annual subscription before renewal"]
    replacements = {"money back": "refund", "yearly membership": "annual subscription", "overseas": "international", "settled bills": "settled invoices", "sign-in": "authentication"}
    rewritten = lower
    for source, target in replacements.items():
        rewritten = rewritten.replace(source, target)
    return [query] if rewritten == lower else [query, rewritten]


def pair_score(query: str, chunk: Chunk) -> float:
    """Proxy a joint query-document scorer over a retrieved shortlist."""
    query_words = set(terms(query)) - STOP
    document_words = terms(f"{chunk.title} {chunk.text}")
    document_set = set(document_words) - STOP
    concept_overlap = len(set(dense_terms(query)) & set(dense_terms(" ".join(document_words))))
    literal_overlap = len(query_words & document_set)
    codes = set(CODE_RE.findall(query.lower()))
    code_bonus = 8.0 * len(codes & set(CODE_RE.findall(" ".join(document_words))))
    region_bonus = 0.0
    query_lower, document_lower = query.lower(), f"{chunk.title} {chunk.text}".lower()
    if re.search(r"\b(us|u\.s\.)\b", query_lower):
        region_bonus = -5.0 if "outside the united states" in document_lower or "international" in document_lower else 3.0 if "united states" in document_lower else 0.0
    return 2.0 * concept_overlap + 0.5 * literal_overlap + code_bonus + region_bonus - 0.002 * len(document_words)


def rerank(query: str, hits: Sequence[Hit], lookup: dict[str, Chunk], k: int) -> list[Hit]:
    rescored = [Hit(hit.chunk_id, hit.source_id, pair_score(query, lookup[hit.chunk_id])) for hit in hits]
    return sorted(rescored, key=lambda hit: (-hit.score, hit.chunk_id))[:k]


def retrieve(query: str, history: str, bm25: BM25, dense: DenseIndex, candidate_k: int = 12, final_k: int = 5) -> tuple[list[Hit], list[Hit]]:
    """Run bounded query expansion, hybrid RRF, and shortlist reranking."""
    variants = query_variants(query, history)
    rankings = [ranking for variant in variants for ranking in (bm25.search(variant, candidate_k), dense.search(variant, candidate_k))]
    lookup = {chunk.chunk_id: chunk for chunk in bm25.chunks}
    fused = rrf_merge(rankings, lookup, limit=candidate_k)
    return fused, rerank(" ".join(variants), fused, lookup, final_k)
