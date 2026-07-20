"""Typed context selection, reduction, isolation, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def token_count(text: str) -> int:
    """Return a transparent whitespace-token proxy for the offline experiment."""
    return len(text.split())


@dataclass(frozen=True)
class Segment:
    kind: str
    trust: str
    content: str
    stable: bool = False
    priority: int = 0
    tags: tuple[str, ...] = ()


ORDER = {"system": 0, "tools": 1, "examples": 2, "retrieved": 3, "history": 4, "query": 5}


def write_context(key: str, value: str) -> Segment:
    """Write a small external reference, not a second memory implementation."""
    return Segment("retrieved", "application", f"REF {key}: {value}", priority=5, tags=(key,))


def select_context(segments: Iterable[Segment], query: str) -> list[Segment]:
    terms = {word.strip(".,?!").lower() for word in query.split() if len(word) > 3}
    return [
        segment
        for segment in segments
        if segment.stable
        or segment.kind == "query"
        or segment.priority >= 8
        or terms.intersection(segment.tags)
    ]


def compress_context(segments: Sequence[Segment], keep_recent: int = 2) -> list[Segment]:
    old, recent = list(segments[:-keep_recent]), list(segments[-keep_recent:])
    if not old:
        return list(segments)
    summary = Segment(
        "history",
        "application-summary",
        f"SUMMARY: {len(old)} earlier low-priority segments omitted; inspect their references if needed.",
        priority=6,
    )
    return [summary, *recent]


def isolate_context(name: str, result: str) -> Segment:
    digest = sum(result.encode("utf-8")) % 100_000
    return Segment(
        "retrieved",
        "tool-observation",
        f"ISOLATED {name}: result_ref={name}:{digest:05d}; summary={result[:72]}",
        priority=6,
        tags=(name,),
    )


def render_context(segments: Iterable[Segment], budget: int) -> str:
    """Render selected segments in one deterministic, trust-labelled order."""
    ordered = sorted(enumerate(segments), key=lambda pair: (ORDER[pair[1].kind], pair[0]))
    rendered: list[str] = []
    used = 0
    for _, segment in ordered:
        block = f'<{segment.kind} trust="{segment.trust}">\n{segment.content}\n</{segment.kind}>'
        size = token_count(block)
        if used + size > budget and not segment.stable and segment.kind != "query":
            continue
        if used + size > budget:
            raise ValueError(f"required {segment.kind} segment exceeds the {budget}-token proxy budget")
        rendered.append(block)
        used += size
    return "\n".join(rendered)
