"""Validate the book's dated landscape database without updating it.

The linter deliberately performs no network requests.  A human checks primary
sources, writes a migration note, and then runs this file as a consistency gate.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


REQUIRED = {
    "id",
    "claim",
    "class",
    "pin",
    "checked",
    "source",
    "owner_chapter",
    "migration_note",
}
CLAIM_CLASSES = {"measured", "vendor-reported", "illustrative"}
OWNER_ANCHOR_PREFIX = "landscape-ref-"
OWNER_ANCHOR_RE = re.compile(r"^landscape-ref-[a-z0-9][a-z0-9-]*$")
QUARTO_ANCHOR_RE = re.compile(r"\{#(landscape-ref-[a-z0-9][a-z0-9-]*)\b[^}]*\}")
HTML_ANCHOR_RE = re.compile(
    r"\bid=[\"'](landscape-ref-[a-z0-9][a-z0-9-]*)[\"']"
)


@dataclass(frozen=True)
class Finding:
    entry_id: str
    field: str
    message: str
    severity: str = "error"

    def render(self) -> str:
        return f"{self.severity.upper():7} {self.entry_id}:{self.field} {self.message}"


def _parse_iso(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _valid_source(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def lint_entries(
    entries: Iterable[Any], *, today: date, max_age_days: int = 120
) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            findings.append(Finding(f"row-{index}", "entry", "must be an object"))
            continue
        entry_id = str(raw.get("id") or f"row-{index}")
        missing = sorted(REQUIRED - raw.keys())
        for field in missing:
            findings.append(Finding(entry_id, field, "is required"))
        if entry_id in seen:
            findings.append(Finding(entry_id, "id", "must be unique"))
        seen.add(entry_id)

        for field in ("claim", "pin", "owner_chapter", "migration_note"):
            value = raw.get(field)
            if field in raw and (not isinstance(value, str) or not value.strip()):
                findings.append(Finding(entry_id, field, "must be a non-empty string"))
        if raw.get("class") not in CLAIM_CLASSES:
            findings.append(
                Finding(entry_id, "class", f"must be one of {sorted(CLAIM_CLASSES)}")
            )
        if "source" in raw and not _valid_source(raw.get("source")):
                findings.append(Finding(entry_id, "source", "must be an https URL"))

        owner_anchor = raw.get("owner_anchor")
        if owner_anchor is not None and (
            not isinstance(owner_anchor, str)
            or not OWNER_ANCHOR_RE.fullmatch(owner_anchor)
            or owner_anchor != f"{OWNER_ANCHOR_PREFIX}{entry_id}"
        ):
            findings.append(
                Finding(
                    entry_id,
                    "owner_anchor",
                    "must equal landscape-ref-<id> and use lowercase letters, digits, and hyphens",
                )
            )

        checked = _parse_iso(raw.get("checked"))
        if checked is None:
            if "checked" in raw:
                findings.append(Finding(entry_id, "checked", "must be ISO YYYY-MM-DD"))
        else:
            age = (today - checked).days
            if age < 0:
                findings.append(Finding(entry_id, "checked", "cannot be in the future"))
            elif age > max_age_days:
                findings.append(
                    Finding(
                        entry_id,
                        "checked",
                        f"is stale by policy ({age} days old; limit {max_age_days})",
                        "warning",
                    )
                )
    return findings


def _registry_anchors(text: str) -> set[str]:
    """Return explicit row-specific anchors from Quarto or raw HTML."""

    return set(QUARTO_ANCHOR_RE.findall(text)) | set(HTML_ANCHOR_RE.findall(text))


def lint_owner_grounding(
    entries: Iterable[Any], *, book_root: Path
) -> list[Finding]:
    """Check registry-to-owner grounding and owner-marker-to-registry ownership."""

    findings: list[Finding] = []
    root = book_root.resolve()
    entry_by_id = {
        raw.get("id"): raw
        for raw in entries
        if isinstance(raw, dict) and isinstance(raw.get("id"), str)
    }
    text_cache: dict[Path, str] = {}

    for entry_id, raw in entry_by_id.items():
        owner = raw.get("owner_chapter")
        if not isinstance(owner, str) or not owner.strip():
            continue
        relative_owner = Path(owner)
        owner_path = (root / relative_owner).resolve()
        try:
            owner_path.relative_to(root)
        except ValueError:
            findings.append(
                Finding(entry_id, "owner_chapter", "must remain inside the book root")
            )
            continue
        if relative_owner.is_absolute() or owner_path.suffix != ".qmd":
            findings.append(
                Finding(entry_id, "owner_chapter", "must be a relative .qmd path")
            )
            continue
        if not owner_path.is_file():
            findings.append(
                Finding(entry_id, "owner_chapter", f"does not exist: {owner}")
            )
            continue
        if owner_path not in text_cache:
            text_cache[owner_path] = owner_path.read_text(encoding="utf-8")
        owner_text = text_cache[owner_path]

        source = raw.get("source")
        source_is_grounded = isinstance(source, str) and source in owner_text
        owner_anchor = raw.get("owner_anchor")
        anchor_is_grounded = (
            isinstance(owner_anchor, str)
            and owner_anchor in _registry_anchors(owner_text)
        )
        if not source_is_grounded and not anchor_is_grounded:
            findings.append(
                Finding(
                    entry_id,
                    "owner_chapter",
                    "must contain the exact source URL or the declared row-specific owner_anchor",
                )
            )

    source_dirs = [root / "chapters", root / "appendices"]
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        for owner_path in source_dir.rglob("*.qmd"):
            text = text_cache.get(owner_path)
            if text is None:
                text = owner_path.read_text(encoding="utf-8")
            relative_owner = owner_path.relative_to(root).as_posix()
            for anchor in _registry_anchors(text):
                entry_id = anchor.removeprefix(OWNER_ANCHOR_PREFIX)
                raw = entry_by_id.get(entry_id)
                if raw is None:
                    findings.append(
                        Finding(
                            entry_id,
                            "owner_anchor",
                            f"marker {anchor} has no registry row",
                        )
                    )
                elif raw.get("owner_anchor") != anchor:
                    findings.append(
                        Finding(
                            entry_id,
                            "owner_anchor",
                            f"marker {anchor} is not declared by its registry row",
                        )
                    )
                elif raw.get("owner_chapter") != relative_owner:
                    findings.append(
                        Finding(
                            entry_id,
                            "owner_chapter",
                            f"marker is in {relative_owner}, not declared owner {raw.get('owner_chapter')}",
                        )
                    )
    return findings


def load_database(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError("database must be an object with an entries array")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--max-age-days", type=int, default=120)
    parser.add_argument(
        "--book-root",
        type=Path,
        help="Book root; defaults to the parent of the database's data directory.",
    )
    parser.add_argument("--warnings-as-errors", action="store_true")
    args = parser.parse_args(argv)

    try:
        database = load_database(args.database)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR   database {exc}")
        return 2
    findings = lint_entries(
        database["entries"], today=args.today, max_age_days=args.max_age_days
    )
    book_root = args.book_root or args.database.resolve().parent.parent
    findings.extend(
        lint_owner_grounding(database["entries"], book_root=book_root)
    )
    for finding in findings:
        print(finding.render())
    errors = sum(item.severity == "error" for item in findings)
    warnings = sum(item.severity == "warning" for item in findings)
    print(
        f"Checked {len(database['entries'])} entries: "
        f"{errors} errors, {warnings} warnings."
    )
    return int(bool(errors or (warnings and args.warnings_as_errors)))


if __name__ == "__main__":
    raise SystemExit(main())
