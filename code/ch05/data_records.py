"""Extract, filter, and exactly decontaminate synthetic WET-style records."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class Document:
    """One extracted record with the provenance fields used by policy."""

    doc_id: str
    url: str
    source: str
    language: str
    rights: str
    text: str


@dataclass(frozen=True)
class FilterPolicy:
    """Auditable thresholds for the lab's transparent quality filter."""

    allowed_rights: tuple[str, ...] = ("licensed", "public-domain", "permission")
    min_words: int = 40
    min_alpha_ratio: float = 0.55
    max_repeated_line_fraction: float = 0.45


def extract_wet(path: Path) -> list[Document]:
    """Extract text and selected headers from the fixture's WET subset."""

    documents: list[Document] = []
    for raw_record in path.read_bytes().split(b"WARC/1.0")[1:]:
        raw_headers, separator, raw_body = raw_record.lstrip(b"\r\n").partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for line in raw_headers.decode("utf-8", errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().casefold()] = value.strip()
        length = int(headers.get("content-length", len(raw_body)))
        body = raw_body[:length].decode("utf-8", errors="replace")
        url = headers.get("warc-target-uri", "unknown://record")
        doc_id = headers.get("warc-record-id") or hashlib.sha1(url.encode()).hexdigest()[:16]
        documents.append(
            Document(
                doc_id=doc_id.strip("<>"),
                url=url,
                source=headers.get("x-source", "unknown"),
                language=headers.get("warc-identified-content-language", "und"),
                rights=headers.get("x-rights", "unknown"),
                text=body.strip(),
            )
        )
    return documents


def quality_features(text: str) -> dict[str, float]:
    """Measure interpretable text features without treating them as truth."""

    words = WORD_RE.findall(text)
    visible = [character for character in text if not character.isspace()]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated = 0.0 if not lines else 1.0 - len(set(lines)) / len(lines)
    alpha = sum(character.isalpha() for character in visible) / max(1, len(visible))
    return {"words": float(len(words)), "alpha_ratio": alpha, "repeated_line_fraction": repeated}


def quality_score(text: str) -> float:
    """Return a bounded score used only to choose a cluster exemplar."""

    features = quality_features(text)
    length_credit = min(1.0, math.log1p(features["words"]) / math.log(2_000))
    return (
        0.45 * features["alpha_ratio"]
        + 0.35 * (1.0 - features["repeated_line_fraction"])
        + 0.20 * length_credit
    )


def filter_documents(
    documents: Iterable[Document], policy: FilterPolicy = FilterPolicy()
) -> tuple[list[Document], list[dict[str, str]]]:
    """Apply rights and quality gates, returning accepted records and reasons."""

    accepted: list[Document] = []
    rejected: list[dict[str, str]] = []
    for document in documents:
        features = quality_features(document.text)
        reason = None
        if document.rights not in policy.allowed_rights:
            reason = "rights-policy"
        elif features["words"] < policy.min_words:
            reason = "too-short"
        elif features["alpha_ratio"] < policy.min_alpha_ratio:
            reason = "low-alpha-ratio"
        elif features["repeated_line_fraction"] > policy.max_repeated_line_fraction:
            reason = "line-repetition"
        if reason is None:
            accepted.append(document)
        else:
            rejected.append({"doc_id": document.doc_id, "reason": reason})
    return accepted, rejected


def _normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.casefold()))


def decontaminate(
    documents: Iterable[Document], evaluation_items: Iterable[str], *, min_chars: int = 24
) -> tuple[list[Document], list[dict[str, str]]]:
    """Remove records containing normalized exact evaluation substrings."""

    needles = [
        normalized
        for item in evaluation_items
        if len(normalized := _normalize(item)) >= min_chars
    ]
    kept: list[Document] = []
    rejected: list[dict[str, str]] = []
    for document in documents:
        normalized = _normalize(document.text)
        match = next((needle for needle in needles if needle in normalized), None)
        if match is None:
            kept.append(document)
        else:
            rejected.append(
                {"doc_id": document.doc_id, "reason": "evaluation-substring", "match": match}
            )
    return kept, rejected
