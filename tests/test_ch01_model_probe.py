"""Focused executable checks for the Chapter 1 model probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch01"))

from model_probe import FixtureClient, run_probe  # noqa: E402


def test_fixture_repeats_for_the_same_controls() -> None:
    client = FixtureClient()
    first = client.complete("Give one short release-readiness reminder.", 0.7, 4)
    second = client.complete("Give one short release-readiness reminder.", 0.7, 4)
    assert first.text == second.text
    assert first.token_scores == second.token_scores


def test_temperature_zero_uses_one_completion() -> None:
    client = FixtureClient()
    outputs = {client.complete("Give one short release-readiness reminder.", 0.0, seed).text for seed in range(20)}
    assert len(outputs) == 1


def test_nonzero_temperature_exposes_more_of_the_distribution() -> None:
    client = FixtureClient()
    outputs = {client.complete("Give one short release-readiness reminder.", 1.2, seed).text for seed in range(20)}
    assert len(outputs) > 1


def test_scores_align_with_fixture_tokens() -> None:
    result = FixtureClient().complete("What did the 1912 Alpine Berry Treaty establish?", 0.0, 0)
    assert result.output_tokens == len(result.token_scores)
    assert [item.token for item in result.token_scores] == result.text.split()


def test_probe_writes_data_figure_and_memo(tmp_path: Path) -> None:
    unique = run_probe(FixtureClient(), tmp_path)
    assert unique[0.0] == 1
    assert unique[1.2] > unique[0.0]
    assert (tmp_path / "sampling-diversity.svg").read_text(encoding="utf-8").startswith("<?xml")
    assert "Least-autonomous baseline" in (tmp_path / "agency-budget-template.md").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "confabulation.json").read_text(encoding="utf-8"))
    assert "not probabilities" in report["warning"]
