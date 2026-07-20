"""A screenshot-bound GUI proposal gate for Chapter 29."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Element:
    element_id: str
    label: str
    box: tuple[int, int, int, int]
    destructive: bool = False


@dataclass(frozen=True)
class Screen:
    revision: int
    width: int
    height: int
    elements: tuple[Element, ...]

    @property
    def digest(self) -> str:
        payload = {
            "revision": self.revision,
            "elements": [element.__dict__ for element in self.elements],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ClickProposal:
    screen_digest: str
    element_id: str
    x: int
    y: int
    destructive: bool


def initial_screen() -> Screen:
    return Screen(
        1,
        1000,
        700,
        (
            Element("save", "Save draft", (700, 610, 820, 670)),
            Element("delete", "Delete account", (830, 610, 980, 670), True),
            Element("email", "Email address", (120, 180, 620, 230)),
        ),
    )


def ground(screen: Screen, label: str) -> ClickProposal:
    """Stand in for a VLM by grounding a label to one exact screen element."""
    matches = [e for e in screen.elements if e.label.casefold() == label.casefold()]
    if len(matches) != 1:
        raise LookupError("label is missing or ambiguous")
    element = matches[0]
    x0, y0, x1, y1 = element.box
    return ClickProposal(
        screen.digest,
        element.element_id,
        (x0 + x1) // 2,
        (y0 + y1) // 2,
        element.destructive,
    )


def authorize(screen: Screen, proposal: ClickProposal, confirmed: bool = False) -> str:
    """Revalidate freshness, target geometry, and destructive confirmation."""
    if proposal.screen_digest != screen.digest:
        return "deny:stale_screen"
    element = next((e for e in screen.elements if e.element_id == proposal.element_id), None)
    if element is None:
        return "deny:missing_target"
    x0, y0, x1, y1 = element.box
    if not (x0 <= proposal.x <= x1 and y0 <= proposal.y <= y1):
        return "deny:coordinates_outside_target"
    if element.destructive and not confirmed:
        return "review:confirmation_required"
    return "allow"


def mutate(screen: Screen) -> Screen:
    """Move controls so an old coordinate proposal becomes stale."""
    moved = tuple(
        replace(element, box=(40, 610, 190, 670))
        if element.element_id == "delete"
        else element
        for element in screen.elements
    )
    return replace(screen, revision=screen.revision + 1, elements=moved)


def run_demo() -> dict[str, object]:
    screen = initial_screen()
    safe = ground(screen, "Save draft")
    destructive = ground(screen, "Delete account")
    changed = mutate(screen)
    return {
        "safe_decision": authorize(screen, safe),
        "destructive_without_confirmation": authorize(screen, destructive),
        "destructive_with_confirmation": authorize(screen, destructive, True),
        "stale_decision": authorize(changed, destructive, True),
        "image_tokens_per_step": 1088,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
