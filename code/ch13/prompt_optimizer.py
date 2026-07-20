"""Finite, measurable prompt optimization for the Chapter 13 lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Ticket:
    text: str
    expected: str


GOLDEN_SET = (
    Ticket("Where is order 41? The tracking page has not moved.", "status"),
    Ticket("The desktop app crashes on launch.", "technical"),
    Ticket("Please refund the duplicate charge.", "refund"),
    Ticket("My package says delivered, but it is not here.", "status"),
    Ticket("The tracking page crashes whenever I open it.", "technical"),
    Ticket("I want my money back because the parcel is late.", "refund"),
    Ticket("Reset links always return a server error.", "technical"),
    Ticket("Has order 77 shipped yet?", "status"),
    Ticket("Cancel and refund the order that has not shipped.", "refund"),
    Ticket("The delivery screen freezes after sign-in.", "technical"),
    Ticket("When should the replacement arrive?", "status"),
    Ticket("Reverse the payment; the app also logged me out.", "refund"),
)


BASELINE_PROMPT = "Classify the customer ticket. Return one label."
CANDIDATE_PROMPTS = (
    BASELINE_PROMPT,
    "You are a helpful support expert. Classify the ticket and be terse.",
    "Return exactly one of: refund, status, technical.",
    (
        "Return exactly one of: refund, status, technical. "
        "Refund means an explicit request to reverse payment. Technical means a "
        "malfunction. Status means a delivery question. PRECEDENCE: refund > "
        "technical > status."
    ),
)


def classify(prompt: str, text: str) -> str:
    """A fixed proxy model whose prompt-sensitive behavior is easy to inspect."""
    words = text.lower()
    refund = any(term in words for term in ("refund", "money back", "reverse the payment"))
    technical = any(term in words for term in ("crash", "error", "freez", "logged me out"))
    status = any(term in words for term in ("where", "tracking", "package", "parcel", "shipped", "arrive", "delivery"))

    if "precedence:" in prompt.lower():
        if refund:
            return "refund"
        if technical:
            return "technical"
        if status:
            return "status"
    if refund:
        return "refund"
    if status:
        return "status"
    return "technical"


def evaluate(prompt: str, examples: Sequence[Ticket] = GOLDEN_SET) -> dict:
    predictions = [classify(prompt, item.text) for item in examples]
    failures = [
        {"text": item.text, "expected": item.expected, "actual": actual}
        for item, actual in zip(examples, predictions)
        if actual != item.expected
    ]
    return {"score": (len(examples) - len(failures)) / len(examples), "failures": failures}


def optimize_prompt(candidates: Sequence[str] = CANDIDATE_PROMPTS) -> dict:
    """Search a finite prompt space against one explicit metric."""
    trials = [{"prompt": prompt, **evaluate(prompt)} for prompt in candidates]
    winner = max(enumerate(trials), key=lambda pair: (pair[1]["score"], -pair[0]))[1]
    return {"baseline": trials[0], "winner": winner, "trials": trials}
