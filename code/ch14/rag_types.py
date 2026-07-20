"""Versioned corpus types and deterministic chunking for Chapter 14."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CODE_RE = re.compile(r"\b[a-z]\d{3,}\b")
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what", "when", "with"}
CANON = {
    "money": "refund", "refunds": "refund", "refunded": "refund", "return": "refund", "returns": "refund",
    "buying": "purchase", "bought": "purchase", "yearly": "annual", "membership": "subscription",
    "plans": "subscription", "plan": "subscription", "renewing": "renewal", "renew": "renewal",
    "stop": "cancel", "overseas": "international", "parcel": "package", "shipment": "package",
    "bills": "invoice", "invoices": "invoice", "download": "export", "erased": "deletion", "erase": "deletion",
    "delete": "deletion", "admins": "administrator", "administrators": "administrator",
    "laptop": "computer", "client": "computer", "login": "authentication", "sign-in": "authentication",
    "wrong": "skew", "differs": "skew", "time": "clock", "locked": "deadlock", "lock": "deadlock",
    "recover": "retry", "recovery": "reset", "severity-one": "priority", "damaged": "damage",
    "undo": "rollback", "bad": "failed", "weekend": "support", "phone": "telephone",
    "us": "united-states", "united": "united-states", "states": "united-states",
    "sign-on": "sso", "sso": "sso", "setting": "configure", "setup": "configure",
}


def terms(text: str) -> list[str]:
    """Tokenize lower-case words and code-like identifiers."""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Document:
    source_id: str
    family: str
    title: str
    version: str
    active: bool
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    title: str
    version: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source_id: str
    score: float


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSONL rows."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_documents(path: Path) -> list[Document]:
    """Load the versioned corpus records."""
    return [Document(**row) for row in load_jsonl(path)]


def chunk_documents(documents: Sequence[Document], size: int, overlap: int) -> list[Chunk]:
    """Chunk active documents with stable, content-derived identities."""
    if size < 8 or not 0 <= overlap < size:
        raise ValueError("require size >= 8 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    step = size - overlap
    for document in documents:
        if not document.active:
            continue
        words = document.text.split()
        for ordinal, start in enumerate(range(0, len(words), step)):
            body = " ".join(words[start : start + size])
            if not body:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()[:8]
            chunk_id = f"{document.source_id}:{ordinal}:{digest}"
            chunks.append(Chunk(chunk_id, document.source_id, document.title, document.version, ordinal, body))
            if start + size >= len(words):
                break
    return chunks
