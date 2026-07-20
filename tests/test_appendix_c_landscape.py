import importlib.util
import json
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "code" / "appc" / "landscape_lint.py"
SPEC = importlib.util.spec_from_file_location("landscape_lint", MODULE_PATH)
assert SPEC and SPEC.loader
landscape_lint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = landscape_lint
SPEC.loader.exec_module(landscape_lint)


def valid_entry(**updates):
    entry = {
        "id": "protocol-example",
        "claim": "An explicitly scoped claim.",
        "class": "vendor-reported",
        "pin": "1.2.3",
        "checked": "2026-07-19",
        "source": "https://example.org/spec/1.2.3",
        "owner_chapter": "chapters/19-protocols-frameworks.qmd",
        "migration_note": "Run compatibility fixtures before upgrading.",
    }
    entry.update(updates)
    return entry


def test_checked_in_database_is_clean():
    data = json.loads((ROOT / "data" / "landscape-2026.json").read_text())
    findings = landscape_lint.lint_entries(
        data["entries"], today=date(2026, 7, 19), max_age_days=120
    )
    findings += landscape_lint.lint_owner_grounding(
        data["entries"], book_root=ROOT
    )
    assert findings == []
    assert len(data["entries"]) == 55


def test_every_registry_owner_is_grounded_in_a_real_book_source():
    data = json.loads((ROOT / "data" / "landscape-2026.json").read_text())
    assert landscape_lint.lint_owner_grounding(data["entries"], book_root=ROOT) == []


def test_owner_needs_exact_source_or_row_specific_anchor(tmp_path):
    owner = tmp_path / "chapters" / "owner.qmd"
    owner.parent.mkdir()
    owner.write_text("# Owner\n\nNo registry evidence here.\n", encoding="utf-8")
    entry = valid_entry(owner_chapter="chapters/owner.qmd")
    findings = landscape_lint.lint_owner_grounding([entry], book_root=tmp_path)
    assert any(item.field == "owner_chapter" for item in findings)

    anchor = "landscape-ref-protocol-example"
    owner.write_text(f"# Owner\n\n[Registry pin]{{#{anchor}}}\n", encoding="utf-8")
    anchored = valid_entry(
        owner_chapter="chapters/owner.qmd", owner_anchor=anchor
    )
    assert landscape_lint.lint_owner_grounding([anchored], book_root=tmp_path) == []


def test_unregistered_owner_marker_is_an_error(tmp_path):
    owner = tmp_path / "appendices" / "owner.qmd"
    owner.parent.mkdir()
    owner.write_text(
        "# Owner\n\n[Orphan control]"
        "{#landscape-ref-not-in-the-registry}\n",
        encoding="utf-8",
    )
    findings = landscape_lint.lint_owner_grounding([], book_root=tmp_path)
    assert any("has no registry row" in item.message for item in findings)


def test_missing_pin_is_an_error():
    entry = valid_entry()
    del entry["pin"]
    findings = landscape_lint.lint_entries([entry], today=date(2026, 7, 19))
    assert any(item.field == "pin" and item.severity == "error" for item in findings)


def test_stale_entry_is_a_warning():
    entry = valid_entry(checked="2025-01-01")
    findings = landscape_lint.lint_entries(
        [entry], today=date(2026, 7, 19), max_age_days=120
    )
    assert [item.severity for item in findings] == ["warning"]


def test_future_and_invalid_dates_fail():
    future = valid_entry(id="future", checked="2026-07-20")
    invalid = valid_entry(id="invalid", checked="07/19/2026")
    findings = landscape_lint.lint_entries(
        [future, invalid], today=date(2026, 7, 19)
    )
    assert {(item.entry_id, item.field) for item in findings} == {
        ("future", "checked"),
        ("invalid", "checked"),
    }


def test_duplicate_ids_and_non_https_sources_fail():
    first = valid_entry(source="http://example.org")
    second = valid_entry()
    findings = landscape_lint.lint_entries([first, second], today=date(2026, 7, 19))
    assert any(item.field == "source" for item in findings)
    assert any(item.field == "id" and "unique" in item.message for item in findings)


def test_cli_exit_codes(tmp_path):
    owner = tmp_path / "chapters" / "19-protocols-frameworks.qmd"
    owner.parent.mkdir()
    owner.write_text(
        "# Fixture\n\nhttps://example.org/spec/1.2.3\n", encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    clean = data_dir / "clean.json"
    clean.write_text(json.dumps({"entries": [valid_entry()]}), encoding="utf-8")
    assert landscape_lint.main([str(clean), "--today", "2026-07-19"]) == 0

    stale = data_dir / "stale.json"
    stale.write_text(
        json.dumps({"entries": [valid_entry(checked="2025-01-01")]}),
        encoding="utf-8",
    )
    assert (
        landscape_lint.main(
            [
                str(stale),
                "--today",
                "2026-07-19",
                "--warnings-as-errors",
            ]
        )
        == 1
    )
