"""Bottom-k shingling, MinHash/LSH candidates, and exact confirmation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from data_records import Document, WORD_RE, quality_score


PRIME = 4_294_967_311


def _shingles(text: str, width: int = 5, limit: int = 512) -> set[str]:
    words = WORD_RE.findall(text.casefold())
    values = {
        " ".join(words[index : index + width])
        for index in range(max(1, len(words) - width + 1))
    }
    ranked = sorted(
        values,
        key=lambda value: (hashlib.blake2b(value.encode(), digest_size=8).digest(), value),
    )
    return set(ranked[:limit])


def _signature(shingles: set[str], permutations: int) -> tuple[int, ...]:
    if not shingles:
        return (PRIME,) * permutations
    hashes = [
        int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") % PRIME
        for value in shingles
    ]
    signature = []
    for index in range(permutations):
        a = 2 * index + 1
        b = 2_654_435_761 * (index + 1)
        signature.append(min((a * value + b) % PRIME for value in hashes))
    return tuple(signature)


def near_deduplicate(
    documents: Iterable[Document],
    *,
    threshold: float = 0.82,
    permutations: int = 32,
    bands: int = 8,
) -> tuple[list[Document], list[dict[str, object]]]:
    """Cluster LSH candidates only after sampled-shingle Jaccard confirmation."""

    rows = list(documents)
    if not 0 < threshold <= 1 or permutations % bands:
        raise ValueError("threshold must be in (0, 1] and bands must divide permutations")
    shingle_sets = [_shingles(document.text) for document in rows]
    signatures = [_signature(shingles, permutations) for shingles in shingle_sets]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    width = permutations // bands
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, signature in enumerate(signatures):
        for band in range(bands):
            buckets[(band, signature[band * width : (band + 1) * width])].append(index)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            candidates.update((left, right) for right in members[offset + 1 :])
    for left, right in sorted(candidates):
        union_size = len(shingle_sets[left] | shingle_sets[right])
        similarity = len(shingle_sets[left] & shingle_sets[right]) / max(1, union_size)
        if similarity >= threshold:
            union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        clusters[find(index)].append(index)
    kept_indices: list[int] = []
    audit: list[dict[str, object]] = []
    for cluster_id, members in enumerate(sorted(clusters.values(), key=lambda value: value[0])):
        exemplar = max(
            members,
            key=lambda index: (quality_score(rows[index].text), len(rows[index].text), -index),
        )
        kept_indices.append(exemplar)
        for member in members:
            union_size = len(shingle_sets[member] | shingle_sets[exemplar])
            similarity = len(shingle_sets[member] & shingle_sets[exemplar]) / max(1, union_size)
            audit.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "member_id": rows[member].doc_id,
                    "kept_id": rows[exemplar].doc_id,
                    "kept": member == exemplar,
                    "jaccard_to_kept": round(similarity, 6),
                }
            )
    return [rows[index] for index in sorted(kept_indices)], audit
