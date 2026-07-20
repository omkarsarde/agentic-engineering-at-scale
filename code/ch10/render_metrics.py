"""Render Chapter 10's deterministic scheduling, quantization, and TTC figure."""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

from serving_lab import run_experiment


WIDTH, HEIGHT = 1140, 380


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start",
         weight: str = "normal") -> str:
    """Return one print-safe SVG text element."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="#111">{escape(value)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, dash: str = "",
         width: float = 1.5) -> str:
    """Return one SVG line."""
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#111" stroke-width="{width}"{pattern}/>' )


def polyline(points: list[tuple[float, float]], dash: str = "", width: float = 2) -> str:
    """Return one unfilled SVG polyline."""
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (f'<polyline points="{coordinates}" fill="none" stroke="#111" '
            f'stroke-width="{width}"{pattern}/>' )


def circle(x: float, y: float, radius: float = 4, fill: str = "white") -> str:
    """Return one circle marker."""
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" '
            'stroke="#111" stroke-width="1.5"/>')


def square(x: float, y: float, radius: float = 4) -> str:
    """Return one square marker."""
    return (f'<rect x="{x-radius:.1f}" y="{y-radius:.1f}" width="{2*radius}" '
            f'height="{2*radius}" fill="#777" stroke="#111"/>')


def vertical_text(x: float, y: float, value: str, size: int = 11) -> str:
    """Return a centered vertical axis label."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(-90 {x:.1f} {y:.1f})" '
            f'font-family="Arial,sans-serif" font-size="{size}" text-anchor="middle" '
            f'fill="#111">{escape(value)}</text>')


def axes(parts: list[str], left: float, top: float, width: float, height: float,
         title: str, xlabel: str, ylabel: str) -> None:
    """Append a shared panel frame and labels."""
    parts.extend([text(left, top - 16, title, 14, weight="bold"),
                  line(left, top, left, top + height, width=1),
                  line(left, top + height, left + width, top + height, width=1),
                  text(left + width / 2, top + height + 35, xlabel, 11, "middle"),
                  vertical_text(left - 40, top + height / 2, ylabel)])


def render(output: Path) -> None:
    """Render all three panels from the shared experiment record."""
    metrics = run_experiment()
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
             f'viewBox="0 0 {WIDTH} {HEIGHT}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<title>Deterministic serving trade-offs</title>']

    # Panel A: capacity use and SLO-qualified goodput share a normalized axis.
    left, top, width, height = 60, 65, 285, 230
    axes(parts, left, top, width, height, "A  Scheduling under one arrival burst",
         "policy", "fraction / requests per step")
    scheduling = metrics["scheduling"]
    values = [("fixed", float(scheduling["static"]["utilization"]),
               float(scheduling["static_goodput"]["requests_per_step"])),
              ("continuous", float(scheduling["continuous"]["utilization"]),
               float(scheduling["continuous_goodput"]["requests_per_step"]))]
    ymap = lambda value: top + height - value * height
    parts.extend([text(left - 8, top + 4, "1.0", 10, "end"),
                  text(left - 8, top + height + 4, "0", 10, "end")])
    for index, (name, utilization, qualified) in enumerate(values):
        center = left + 85 + index * 130
        for shift, value, fill in ((-18, utilization, "#777"), (18, qualified, "white")):
            y = ymap(value)
            parts.append(f'<rect x="{center+shift-12:.1f}" y="{y:.1f}" width="24" '
                         f'height="{top+height-y:.1f}" fill="{fill}" stroke="#111"/>')
            parts.append(text(center + shift, y - 6, f"{value:.2f}", 9, "middle"))
        parts.append(text(center, top + height + 18, name, 10, "middle"))
    parts.extend([text(left + 8, top + 18, "solid: slot utilization", 10),
                  text(left + 8, top + 34, "open: SLO goodput", 10)])

    # Panel B: finer blocks buy lower error with scale metadata.
    left, top, width, height = 430, 65, 285, 230
    axes(parts, left, top, width, height, "B  Four-bit block quantization",
         "block size (values)", "RMSE (fixture units)")
    rows = list(metrics["quantization"])
    xmap = lambda value: left + (value - 16) / (256 - 16) * width
    ymap = lambda value: top + height - value / 1.6 * height
    points = [(xmap(float(row["block_size"])), ymap(float(row["rmse"]))) for row in rows]
    parts.extend([polyline(points), text(left - 8, top + 4, "1.6", 10, "end"),
                  text(left - 8, top + height + 4, "0", 10, "end")])
    for row, (x, y) in zip(rows, points):
        parts.extend([circle(x, y, 5), text(x, top + height + 18, str(row["block_size"]), 10, "middle"),
                      text(x, y - 10, f'{float(row["rmse"]):.2f}', 9, "middle")])
    parts.extend([text(left + 7, top + 18, "effective bits/value:", 10),
                  text(left + 7, top + 34, "5.00, 4.25, 4.06", 10)])

    # Panel C: quality saturates while tokens continue to accumulate.
    left, top, width, height = 800, 65, 285, 230
    axes(parts, left, top, width, height, "C  Test-time compute budget",
         "reasoning-token budget", "synthetic task score")
    rows = list(metrics["reasoning_budget"])
    xmap = lambda value: left + (value - 32) / (512 - 32) * width
    ymap = lambda value: top + height - (value - 0.58) / (0.80 - 0.58) * height
    points = [(xmap(float(row["reasoning_tokens"])), ymap(float(row["synthetic_score"]))) for row in rows]
    parts.extend([polyline(points), text(left - 8, top + 4, "0.80", 10, "end"),
                  text(left - 8, top + height + 4, "0.58", 10, "end")])
    for row, (x, y) in zip(rows, points):
        parts.extend([square(x, y, 4),
                      text(x, top + height + 18, str(row["reasoning_tokens"]), 9, "middle")])
    parts.extend([line(xmap(256), top, xmap(256), top + height, "4 4", 1),
                  text(xmap(256) + 5, top + 17, "diminishing return", 10)])

    parts.append(text(WIDTH / 2, HEIGHT - 10,
                      "All panels are analytical or synthetic deterministic fixtures, not deployed-model or accelerator measurements.",
                      10, "middle"))
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("serving_metrics.svg")
    render(destination)
