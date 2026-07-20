"""Decision-preserving history compaction and prefix-reuse measurement."""

from __future__ import annotations

from typing import Sequence

from context_assembly import token_count


DECISIONS = (
    "DECISION D1: preserve typed tool results because correlation IDs are evidence.",
    "DECISION D2: retry at most twice because duplicate writes are unsafe.",
    "DECISION D3: pin schema version v4 because replay depends on it.",
)
OPEN_QUESTIONS = (
    "OPEN Q1: reproduce the timeout under the production proxy.",
    "OPEN Q2: confirm whether order writes are idempotent.",
)


def synthetic_history() -> list[str]:
    """Create a 40-message debugging trace with sparse high-value state."""
    anchors = {5: DECISIONS[0], 11: OPEN_QUESTIONS[0], 18: DECISIONS[1], 24: "ERROR E1: proxy closed the stream before the tool result.", 30: OPEN_QUESTIONS[1], 34: DECISIONS[2]}
    return [
        anchors.get(turn, f"TURN {turn:02d}: inspected routine trace batch and recorded no state change " + "detail " * 10).strip()
        for turn in range(1, 41)
    ]


def compact_history(messages: Sequence[str], budget: int, trigger: float = 0.65, keep_recent: int = 4) -> tuple[list[str], bool]:
    """Classify durable state before replacing expendable middle history."""
    if token_count("\n".join(messages)) <= budget * trigger:
        return list(messages), False
    durable = [message for message in messages if message.startswith(("DECISION ", "OPEN ", "ERROR "))]
    recent = list(messages[-keep_recent:])
    omitted = len(messages) - len({*durable, *recent})
    summary = f"SUMMARY: {omitted} routine messages omitted after classification; no new decision recorded."
    compacted = [*durable, summary, *[message for message in recent if message not in durable]]
    assert_survival(messages, compacted)
    return compacted, True


def assert_survival(before: Sequence[str], after: Sequence[str]) -> None:
    joined = "\n".join(after)
    required = [message for message in before if message.startswith(("DECISION ", "OPEN "))]
    missing = [message for message in required if message not in joined]
    if missing:
        raise AssertionError(f"compaction lost durable state: {missing}")


def common_prefix_bytes(left: str, right: str) -> int:
    """Count identical leading bytes; the first differing byte ends reuse."""
    a, b = left.encode("utf-8"), right.encode("utf-8")
    for index, (one, two) in enumerate(zip(a, b)):
        if one != two:
            return index
    return min(len(a), len(b))


def cache_cost_example() -> dict:
    """Illustrative unit economics chosen to expose the timestamp multiplier."""
    stable_tokens, volatile_tokens, read_rate = 7_000, 300, 0.10
    hit = stable_tokens * read_rate + volatile_tokens
    miss = stable_tokens + volatile_tokens
    return {"stable_prefix_cost": hit, "broken_prefix_cost": miss, "multiplier": miss / hit}


def simulate_turns(compaction: bool, stable_prefix: bool, turns: int = 50) -> list[dict]:
    """Run a growing transcript and record context and exact-prefix observables."""
    policy = "SYSTEM\n" + "stable-policy-token " * 7_000
    history: list[str] = []
    previous = ""
    ledger: list[dict] = []
    budget = 12_000
    for turn in range(1, turns + 1):
        history.append(f"TURN {turn:02d} observation " + "working-token " * 105)
        did_compact = False
        body = policy + "\n" + "\n".join(history)
        if compaction and token_count(body) > budget * 0.65:
            history = [f"COMPACTED THROUGH TURN {turn - 3:02d} decision-ledger unchanged " + "summary-token " * 90, *history[-3:]]
            body = policy + "\n" + "\n".join(history)
            did_compact = True
        prompt = body if stable_prefix else f"NOW=2026-07-19T12:00:{turn:02d}Z\n{body}"
        reusable = common_prefix_bytes(previous, prompt) if previous else 0
        ledger.append(
            {
                "scenario": f"compaction={'on' if compaction else 'off'}, prefix={'stable' if stable_prefix else 'broken'}",
                "turn": turn,
                "tokens": token_count(prompt),
                "utilization": token_count(prompt) / budget,
                "cache_hit_rate": reusable / len(prompt.encode("utf-8")),
                "compacted": did_compact,
            }
        )
        previous = prompt
    return ledger
