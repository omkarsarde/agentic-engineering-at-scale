"""Generate the synthetic WET fixture and disjoint training/evaluation text."""

from __future__ import annotations

import random
from pathlib import Path

from data_records import Document
from experiment_fixture import EVALUATION_NEEDLE, FACTS, HOLDOUT_FACTS


def _clean_body(record: int, body_bytes: int, *, contaminate: bool = False) -> str:
    lines = [EVALUATION_NEEDLE] if contaminate else []
    index = 0
    while len("\n".join(lines).encode("utf-8")) < body_bytes:
        fact = FACTS[(record + index) % len(FACTS)]
        lines.append(
            f"Archive entry {record:04d}-{index:05d} preserves a checked statement. "
            f"{fact} The editor records its source and keeps the wording readable."
        )
        index += 1
    return "\n".join(lines)


def _repeat_to_bytes(line: str, body_bytes: int) -> str:
    repetitions = max(2, body_bytes // max(1, len((line + "\n").encode("utf-8"))))
    return (line + "\n") * repetitions


def write_synthetic_wet(
    path: Path, *, target_bytes: int = 50 * 1024 * 1024, body_bytes: int = 64 * 1024
) -> int:
    """Write licensed, duplicated, restricted, noisy, and short fixture records."""

    if target_bytes < body_bytes * 10:
        raise ValueError("target_bytes must hold at least one complete ten-record group")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_clean = ""
    records = 0
    with path.open("wb") as handle:
        while handle.tell() < target_bytes:
            slot = records % 10
            group = records // 10
            source, language, rights = "web", "en", "licensed"
            if slot == 0:
                source = "reference"
                previous_clean = _clean_body(records, body_bytes, contaminate=group == 0)
                body = previous_clean
            elif slot == 1:
                source = "reference"
                body = previous_clean + f"\nRevision {group} preserves the same evidence."
            elif slot == 2:
                body = _repeat_to_bytes("BUY NOW CLICK HERE LIMITED OFFER", body_bytes)
            elif slot == 3:
                rights = "restricted"
                body = _clean_body(records, body_bytes)
            elif slot == 4:
                body = _repeat_to_bytes("### 0000 !!! --- 9999 ???", body_bytes)
            elif slot == 5:
                source, rights = "library", "public-domain"
                previous_clean = _clean_body(records, body_bytes)
                body = previous_clean
            elif slot == 6:
                rights = "permission"
                body = _repeat_to_bytes("ARCHIVE INDEX ARCHIVE INDEX ARCHIVE INDEX", body_bytes)
            elif slot == 7:
                rights = "permission"
                body = "Short notice."
            elif slot == 8:
                rights = "permission"
                body = _repeat_to_bytes("MENU LOGIN COOKIE HOME MENU LOGIN COOKIE HOME", body_bytes)
            else:
                source, rights = "community", "permission"
                body = _clean_body(records, body_bytes)
            encoded = body.encode("utf-8")
            headers = (
                "WARC/1.0\r\n"
                f"WARC-Record-ID: <synthetic-{records:05d}>\r\n"
                f"WARC-Target-URI: https://fixture.invalid/{records:05d}\r\n"
                f"WARC-Identified-Content-Language: {language}\r\n"
                f"X-Source: {source}\r\n"
                f"X-Rights: {rights}\r\n"
                f"Content-Length: {len(encoded)}\r\n\r\n"
            ).encode("utf-8")
            handle.write(headers)
            handle.write(encoded)
            handle.write(b"\r\n\r\n")
            records += 1
    return records


def sample_text(
    documents: list[Document], *, seed: int, characters_per_doc: int = 3_000
) -> str:
    """Return a seeded fixed-size excerpt from each shuffled document."""

    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    return "\n".join(document.text[:characters_per_doc] for document in shuffled)


def make_holdout_text(records: int = 320) -> str:
    """Create independently worded facts never placed in the training fixture."""

    return "\n".join(
        f"Field report 9000-{index:05d} was independently reviewed. "
        f"{HOLDOUT_FACTS[index % len(HOLDOUT_FACTS)]} "
        "A curator preserved provenance and legible prose."
        for index in range(records)
    )
