"""Regression tests for the manuscript structure validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "newbook_structure_validator", ROOT / "scripts" / "validate_book.py"
)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def test_local_link_validator_checks_fragments(tmp_path: Path) -> None:
    """A present file with a misspelled section anchor is still a broken link."""
    target = tmp_path / "target.qmd"
    target.write_text(
        "# Explicit {#sec-explicit}\n\n## Automatic heading\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.qmd"
    source.write_text(
        "[explicit](target.qmd#sec-explicit)\n"
        "[automatic](target.qmd#automatic-heading)\n"
        "[broken](target.qmd#sec-misspelled)\n",
        encoding="utf-8",
    )

    assert VALIDATOR.validate_local_links(source) == [
        "broken local fragment: target.qmd#sec-misspelled"
    ]


def test_chapter_template_cannot_recreate_duplicate_titles() -> None:
    """The starter template delegates numbering to Quarto exactly once."""
    text = (ROOT / "templates" / "chapter-template.qmd").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    assert "title:" not in frontmatter
    assert "# Chapter NN" not in text
    assert "# Exact title [F] {#sec-chNN}" in text
