"""Validate the structural and pedagogical contracts of the Quarto book."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "_quarto.yml"

APPARATUS = {
    "contents",
    "what you need going in",
    "what you will build",
    "build",
    "what endures, what changes",
    "exercises",
    "notes and sources",
}

BANNED = (
    "obviously",
    "of course",
    "clearly",
    "trivially",
    "senior-peer to senior-peer",
    "this is the wrong book",
    "you will not be asked",
    "you already know this from",
)

ROUTE_B_REQUIREMENTS = {
    12: ("Chapter 9", "Chapter 7"),
    13: ("Chapter 10", "Chapter 3"),
    14: ("Chapter 9", "Chapter 11"),
    16: ("Chapter 8",),
    17: ("Chapter 10",),
    23: ("Route B primer",),
    25: ("Route B primer",),
    26: ("Chapter 10",),
    29: ("Chapter 2",),
}


def configured_chapters(text: str) -> list[str]:
    """Return qmd paths from the simple list entries in `_quarto.yml`."""
    return re.findall(r"(?m)^\s*-\s+([A-Za-z0-9_./-]+\.qmd)\s*$", text)


def strip_code(text: str) -> str:
    """Remove fenced code so prose checks do not inspect examples."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def chapter_number(path: Path) -> int | None:
    """Read a two-digit chapter number from a chapter filename."""
    match = re.match(r"(\d{2})-", path.name)
    return int(match.group(1)) if match else None


def h2_titles(text: str) -> list[str]:
    """Return normalized level-two headings."""
    titles: list[str] = []
    for heading in re.findall(r"(?m)^##\s+(.+?)\s*$", strip_code(text)):
        heading = re.sub(r"\s*\{[^}]+\}\s*$", "", heading)
        titles.append(heading.strip())
    return titles


def visual_count(text: str) -> int:
    """Count source-backed figures without counting decorative icons."""
    diagrams = len(re.findall(r"```\{(?:mermaid|dot)\}", text))
    images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", strip_code(text)))
    computed = len(re.findall(r"(?m)^#\|\s*label:\s*fig-", text))
    return diagrams + images + computed


def validate_file(path: Path) -> tuple[list[str], list[str]]:
    """Return errors and warnings for one manuscript source."""
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    prose = strip_code(text)
    number = chapter_number(path)

    if any(marker in text for marker in ("�", "â€”", "â†", "Â§")):
        errors.append("contains likely mojibake")

    if number is not None:
        h1 = re.findall(r"(?m)^#\s+(.+?)\s*$", prose)
        if len(h1) != 1:
            errors.append(f"expected exactly one H1, found {len(h1)}")
        elif re.match(rf"Chapter\s+{number}\b", h1[0]):
            errors.append("H1 repeats the chapter number supplied by Quarto")

        yaml_title = re.search(r'(?m)^title:\s*["\']?(.+?)["\']?\s*$', text)
        if yaml_title:
            errors.append("YAML title duplicates the anchored body H1 in a Quarto book")

        headings = h2_titles(text)
        normalized = [h.casefold() for h in headings]
        counts = Counter(normalized)
        duplicates = [name for name, count in counts.items() if count > 1]
        if duplicates:
            errors.append("duplicate H2 headings: " + ", ".join(duplicates))

        required = {
            "what you need going in",
            "contents",
            "what you will build",
            "build",
            "what endures, what changes",
            "exercises",
        }
        missing = sorted(required - set(normalized))
        if missing:
            errors.append("missing apparatus: " + ", ".join(missing))

        teaching = [
            h for h in normalized
            if h not in APPARATUS and not h.startswith("landscape 2026")
        ]
        if not 6 <= len(teaching) <= 10:
            errors.append(
                f"expected 6–10 teaching H2s, found {len(teaching)}: "
                + ", ".join(teaching)
            )

        figures = visual_count(text)
        if not 2 <= figures <= 5:
            errors.append(f"expected 2–5 figures, found {figures}")

        exercise_match = re.search(
            r"(?ms)^##\s+Exercises.*?(?=^##\s+|\Z)", prose
        )
        exercise_count = 0
        if exercise_match:
            exercise_count = len(
                re.findall(r"(?m)^\s*\d+\.\s+", exercise_match.group(0))
            )
        if not 5 <= exercise_count <= 8:
            errors.append(f"expected 5–8 exercises, found {exercise_count}")

        required_backfill = ROUTE_B_REQUIREMENTS.get(number)
        if required_backfill:
            for phrase in required_backfill:
                if phrase.casefold() not in prose.casefold():
                    errors.append(f"missing route-B obligation mentioning {phrase!r}")

        word_count = len(re.findall(r"\b[\w'-]+\b", prose))
        if word_count < 4000:
            warnings.append(f"short chapter: {word_count} prose words (target 4,000–7,000)")
        elif word_count > 7000:
            warnings.append(f"long chapter: {word_count} prose words (target 4,000–7,000)")

    lower_prose = prose.casefold()
    for phrase in BANNED:
        if re.search(rf"\b{re.escape(phrase)}\b", lower_prose):
            errors.append(f"banned phrase: {phrase!r}")

    if re.search(r"(?<!\\)\$\d", prose):
        errors.append("unescaped currency marker may be parsed as math")

    for block in re.finditer(
        r"(?ms)^:::.*?landscape-2026.*?^:::\s*$", text
    ):
        content = block.group(0)
        if "Verify live:" not in content:
            errors.append("Landscape 2026 block lacks 'Verify live:'")
        if not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", content):
            errors.append("Landscape 2026 block lacks an ISO verification date")

    return errors, warnings


def pandoc_heading_id(heading: str) -> str:
    """Approximate Pandoc's automatic identifier for a Markdown heading."""
    heading = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[`*_~]", "", heading).casefold()
    heading = re.sub(r"[^\w\s.-]", "", heading, flags=re.UNICODE)
    heading = re.sub(r"\s+", "-", heading.strip())
    heading = re.sub(r"^[^\w]*", "", heading, flags=re.UNICODE)
    return heading


def document_anchors(path: Path) -> set[str]:
    """Return explicit and automatic anchors defined by a text document."""
    if path.suffix.casefold() not in {".qmd", ".md", ".markdown", ".html", ".htm"}:
        return set()

    text = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r"\{#[A-Za-z0-9_.:-]+[^}]*\}", text))
    anchors = {item[2:].split()[0].rstrip("}") for item in anchors}
    anchors.update(re.findall(r"\bid=[\"']([^\"']+)[\"']", text))

    automatic_counts: Counter[str] = Counter()
    for raw_heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", strip_code(text)):
        if re.search(r"\{#[A-Za-z0-9_.:-]+[^}]*\}\s*$", raw_heading):
            continue
        raw_heading = re.sub(r"\s*\{[^}]+\}\s*$", "", raw_heading)
        base = pandoc_heading_id(raw_heading)
        if not base:
            continue
        count = automatic_counts[base]
        anchors.add(base if count == 0 else f"{base}-{count}")
        automatic_counts[base] += 1
    return anchors


def validate_local_links(path: Path) -> list[str]:
    """Return unresolved relative files and fragments in Markdown links."""
    text = strip_code(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = unquote(raw_target.strip().strip("<>").split(maxsplit=1)[0])
        if "://" in target or target.startswith(("mailto:", "data:")):
            continue

        file_part, separator, fragment = target.partition("#")
        candidate = path if not file_part else (path.parent / file_part).resolve()
        if not candidate.exists():
            errors.append(f"broken local link: {target}")
            continue

        if separator and fragment:
            if candidate not in anchor_cache:
                anchor_cache[candidate] = document_anchors(candidate)
            anchors = anchor_cache[candidate]
            if anchors and fragment not in anchors:
                errors.append(f"broken local fragment: {target}")
    return errors


def main() -> int:
    """Validate all files named by the Quarto book configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true", help="treat word-count warnings as errors"
    )
    args = parser.parse_args()

    relative_paths = configured_chapters(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    for relative in relative_paths:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"{relative}: missing source file")
            continue
        file_errors, file_warnings = validate_file(path)
        errors.extend(f"{relative}: {message}" for message in file_errors)
        warnings.extend(f"{relative}: {message}" for message in file_warnings)
        errors.extend(f"{relative}: {message}" for message in validate_local_links(path))

    for warning in warnings:
        print(f"WARNING {warning}")
    if args.strict:
        errors.extend(warnings)
    for error in errors:
        print(f"ERROR   {error}")

    print(
        f"Checked {len(relative_paths)} manuscript files: "
        f"{len(errors)} errors, {len(warnings)} warnings."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
