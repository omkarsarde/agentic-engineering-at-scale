"""Focused tests for Chapter 13's context-engineering invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


CODE = Path(__file__).parents[1] / "code" / "ch13"
sys.path.insert(0, str(CODE))

from context_pipeline import (  # noqa: E402
    DECISIONS,
    OPEN_QUESTIONS,
    Segment,
    assert_survival,
    cache_cost_example,
    compact_history,
    common_prefix_bytes,
    optimize_prompt,
    render_context,
    run_build,
    select_context,
    simulate_turns,
    synthetic_history,
)


def test_optimizer_beats_baseline_and_reports_losers() -> None:
    result = optimize_prompt()
    assert result["winner"]["score"] > result["baseline"]["score"]
    assert result["winner"]["score"] == 1.0
    assert len(result["trials"]) == 4
    assert any(trial["failures"] for trial in result["trials"] if trial is not result["winner"])


def test_selection_keeps_stable_query_and_relevant_segments() -> None:
    segments = [
        Segment("system", "developer", "policy", stable=True),
        Segment("retrieved", "tool", "order 41 shipped", tags=("order",)),
        Segment("retrieved", "tool", "weather is clear", tags=("weather",)),
        Segment("query", "user", "Where is my order?"),
    ]
    selected = select_context(segments, "Where is my order?")
    assert [segment.content for segment in selected] == ["policy", "order 41 shipped", "Where is my order?"]


def test_renderer_is_deterministic_and_trust_labelled() -> None:
    segments = [
        Segment("query", "user", "answer this"),
        Segment("system", "developer", "follow policy", stable=True),
    ]
    first = render_context(segments, budget=20)
    second = render_context(segments, budget=20)
    assert first == second
    assert first.index("<system") < first.index("<query")
    assert 'trust="developer"' in first and 'trust="user"' in first


def test_required_segment_cannot_be_silently_dropped() -> None:
    with pytest.raises(ValueError, match="required query"):
        render_context([Segment("query", "user", "one two three four")], budget=2)


def test_compaction_preserves_every_literal_decision_and_open_question() -> None:
    before = synthetic_history()
    after, changed = compact_history(before, budget=500)
    joined = "\n".join(after)
    assert changed
    assert len(after) < len(before)
    assert all(item in joined for item in (*DECISIONS, *OPEN_QUESTIONS))


def test_survival_check_rejects_a_loss() -> None:
    with pytest.raises(AssertionError, match="lost durable state"):
        assert_survival([DECISIONS[0], "routine"], ["routine"])


def test_prefix_reuse_ends_at_first_differing_byte() -> None:
    assert common_prefix_bytes("static\nA", "static\nB") == len("static\n".encode())
    assert common_prefix_bytes("NOW=1\nstatic", "NOW=2\nstatic") < len("NOW=1\nstatic".encode())


def test_cache_cost_fixture_exposes_seven_point_three_multiplier() -> None:
    assert cache_cost_example()["multiplier"] == pytest.approx(7.3)


def test_fifty_turn_grid_separates_compaction_and_prefix_effects() -> None:
    report = run_build()
    rows = report["ledger"]
    assert len(rows) == 4 * 50
    compact_stable = simulate_turns(True, True)
    uncompact_stable = simulate_turns(False, True)
    compact_broken = simulate_turns(True, False)
    assert max(row["utilization"] for row in compact_stable) < 0.70
    assert uncompact_stable[-1]["utilization"] > 1.0
    assert sum(row["cache_hit_rate"] for row in compact_stable[1:]) / 49 > 0.85
    assert sum(row["cache_hit_rate"] for row in compact_broken[1:]) / 49 < 0.01
    assert report["compaction"]["lost_decisions"] == 0
    assert report["compaction"]["lost_open_questions"] == 0
