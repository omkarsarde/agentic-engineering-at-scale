"""CSV, digest, and document-decision provenance for the Chapter 5 build."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from data_records import Document


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a nonempty row sequence with stable field order."""

    if not rows:
        raise ValueError(f"cannot infer columns for empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(text: str) -> str:
    """Hash UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_ledger(
    documents: list[Document],
    filter_rejections: list[dict[str, str]],
    cluster_rows: list[dict[str, object]],
    contamination_rejections: list[dict[str, str]],
    mixed: list[Document],
) -> list[dict[str, object]]:
    """Join provenance to the final decision for every extracted document."""

    filter_map = {row["doc_id"]: row["reason"] for row in filter_rejections}
    cluster_map = {str(row["member_id"]): row for row in cluster_rows}
    contamination_map = {row["doc_id"]: row["reason"] for row in contamination_rejections}
    selected_ids = {document.doc_id for document in mixed}
    ledger: list[dict[str, object]] = []
    for document in documents:
        cluster = cluster_map.get(document.doc_id)
        if document.doc_id in filter_map:
            stage, decision, reason = "filter", "rejected", filter_map[document.doc_id]
        elif cluster is not None and not bool(cluster["kept"]):
            stage, decision, reason = "deduplication", "rejected", "near-duplicate"
        elif document.doc_id in contamination_map:
            stage, decision, reason = "decontamination", "rejected", contamination_map[document.doc_id]
        elif document.doc_id in selected_ids:
            stage, decision, reason = "mixture", "selected", "source-quota"
        else:
            stage, decision, reason = "mixture", "eligible-not-selected", "source-quota"
        ledger.append(
            {
                "doc_id": document.doc_id,
                "url": document.url,
                "source": document.source,
                "language": document.language,
                "rights": document.rights,
                "content_sha256": sha256_text(document.text),
                "policy_version": "ch05-transparent-filter-v1",
                "final_stage": stage,
                "decision": decision,
                "reason": reason,
                "cluster_id": "" if cluster is None else cluster["cluster_id"],
                "kept_id": "" if cluster is None else cluster["kept_id"],
            }
        )
    return ledger
