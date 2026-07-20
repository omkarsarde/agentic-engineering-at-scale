"""Focused multimodal extraction, retrieval, and GUI-gate tests."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch29"
sys.path.insert(0, str(CODE))

from gui_step import authorize, ground, initial_screen, mutate, run_demo  # noqa: E402
from maxsim_demo import run_demo as maxsim_report  # noqa: E402
from page_reader import evaluate, fixture_extract, make_pages, validate_extraction  # noqa: E402


def test_resolution_plateaus_while_tokens_keep_growing() -> None:
    low, medium, high, extreme = [evaluate(value) for value in (1, 4, 9, 16)]
    assert low["field_accuracy"] < medium["field_accuracy"] < high["field_accuracy"]
    assert high["field_accuracy"] == extreme["field_accuracy"]
    assert high["image_tokens_per_page"] < extreme["image_tokens_per_page"]


def test_every_field_carries_in_page_evidence() -> None:
    page = make_pages(1)[0]
    extraction = fixture_extract(page, 4)
    validate_extraction(extraction, page)
    extraction["total"]["box"]["x1"] = page.width + 1
    try:
        validate_extraction(extraction, page)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-page evidence was accepted")


def test_maxsim_resists_the_constructed_mean_flood() -> None:
    report = maxsim_report()
    assert report["pooled"]["flooded"] > report["pooled"]["relevant"]
    assert report["maxsim"]["relevant"] > report["maxsim"]["flooded"]


def test_safe_and_destructive_actions_have_different_gates() -> None:
    report = run_demo()
    assert report["safe_decision"] == "allow"
    assert report["destructive_without_confirmation"] == "review:confirmation_required"
    assert report["destructive_with_confirmation"] == "allow"


def test_stale_screenshot_cannot_authorize_click() -> None:
    screen = initial_screen()
    proposal = ground(screen, "Delete account")
    assert authorize(mutate(screen), proposal, confirmed=True) == "deny:stale_screen"


def test_coordinates_must_still_land_inside_named_target() -> None:
    screen = initial_screen()
    proposal = ground(screen, "Save draft")
    moved = proposal.__class__(proposal.screen_digest, proposal.element_id, 0, 0, False)
    assert authorize(screen, moved) == "deny:coordinates_outside_target"
