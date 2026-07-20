"""Focused session-semantics and latency tests for Chapter 30."""

from __future__ import annotations

import sys
from pathlib import Path


CODE = Path(__file__).parents[1] / "code" / "ch30"
sys.path.insert(0, str(CODE))

from rvq_toy import demo as rvq_demo  # noqa: E402
from voice_loop import (  # noqa: E402
    AudioChunk,
    AudioFrame,
    VoiceSession,
    latency_report,
    run_race_and_tool_fixture,
)


def test_each_rvq_stage_reduces_reconstruction_error() -> None:
    errors = rvq_demo()
    assert all(new < old for old, new in zip(errors, errors[1:]))


def test_silence_closes_turn_only_after_threshold() -> None:
    session = VoiceSession(silence_ms=300)
    assert not session.observe(AudioFrame(0, 0.8))
    assert not session.observe(AudioFrame(299, 0.0))
    assert session.observe(AudioFrame(300, 0.0))


def test_barge_in_clears_queued_audio_and_rejects_late_chunk() -> None:
    session = VoiceSession()
    response_id = session.start_response()
    assert session.accept_chunk(AudioChunk(response_id, 0, "hello"))
    session.observe(AudioFrame(10, 0.9))
    assert session.output == []
    assert not session.accept_chunk(AudioChunk(response_id, 1, "late"))


def test_confirmation_is_bound_to_exact_tool_arguments() -> None:
    report = run_race_and_tool_fixture()
    assert report["mismatched_confirmation"] == "review:confirmation_mismatch"
    assert report["matched_confirmation"] == "allow"
    assert report["effect_count"] == 1


def test_cancelled_response_cannot_commit_effect() -> None:
    session = VoiceSession()
    session.start_response()
    proposal = session.propose_refund("order-1", 15)
    session.cancel_for_barge_in()
    decision = session.confirm_and_execute(proposal, {"order_id": "order-1", "amount": 15})
    assert decision == "deny:cancelled_response"
    assert session.effects == []


def test_streaming_fixture_meets_p50_but_reports_tail() -> None:
    report = latency_report()
    assert report["turns"] == 20
    assert report["sequential_p50_ms"] > 1000
    assert report["overlapped_p50_ms"] < 1000
    assert report["overlapped_p95_ms"] >= report["overlapped_p50_ms"]
