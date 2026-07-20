"""Render the Chapter 9 deterministic roofline, calibration, and abstention figure."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from inference_lab import run_experiment


WIDTH, HEIGHT = 1140, 360


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="#111">{escape(value)}</text>')


def line(x1: float, y1: float, x2: float, y2: float, dash: str = "", width: float = 2) -> str:
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#111" stroke-width="{width}"{style}/>'


def polyline(points: list[tuple[float, float]], dash: str = "", width: float = 2) -> str:
    style = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="#111" stroke-width="{width}"{style}/>'


def circle(x: float, y: float, radius: float = 4, fill: str = "#fff") -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="#111" stroke-width="1.5"/>'


def square(x: float, y: float, radius: float = 4) -> str:
    return f'<rect x="{x-radius:.1f}" y="{y-radius:.1f}" width="{2*radius}" height="{2*radius}" fill="#777" stroke="#111"/>'


def axes(parts: list[str], left: float, top: float, width: float, height: float, title: str, xlabel: str) -> None:
    parts.extend([text(left, top - 13, title, 14, weight="bold"),
                  line(left, top, left, top + height, width=1),
                  line(left, top + height, left + width, top + height, width=1),
                  text(left + width / 2, top + height + 34, xlabel, 11, "middle")])


def render(output: Path) -> None:
    metrics = run_experiment()
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<title>Illustrative request roofline, fixture calibration, and abstention trade-off</title>']

    # Panel A: illustrative roofline.
    left, top, w, h = 55, 55, 300, 235
    axes(parts, left, top, w, h, "A  One-request roofline (illustrative)", "arithmetic intensity (FLOP/byte, log₂)")
    xmap = lambda value: left + math.log2(value) / 11 * w
    ymap = lambda value: top + h - value / 80 * h
    parts.extend([text(left - 8, top + h + 4, "0", 10, "end"), text(left - 8, top + 4, "80", 10, "end"),
                  text(left - 35, top + h / 2, "TFLOP/s", 10, "middle"),
                  polyline([(xmap(1), ymap(2)), (xmap(40), ymap(80)), (xmap(2048), ymap(80))], width=2.5),
                  line(xmap(40), top, xmap(40), top + h, "4 4", 1), text(xmap(40) + 4, top + 16, "ridge = 40", 10)])
    parts.append(circle(xmap(1), ymap(2), 5, "#111"))
    parts.append(text(xmap(1) + 7, ymap(2) - 7, "decode", 10))
    for row in metrics["profiles"]:
        x, y = xmap(float(row["prefill_intensity"])), ymap(80)
        parts.extend([square(x, y, 4), text(x, y - 9, str(row["prompt_tokens"]), 9, "middle")])
    parts.append(text(left + w - 2, top + 31, "prefill prompt tokens", 10, "end"))

    # Panel B: reliability before and after temperature calibration.
    left, top, w, h = 430, 55, 300, 235
    axes(parts, left, top, w, h, "B  Reliability on synthetic QA fixture", "reported confidence")
    xmap = lambda value: left + value * w
    ymap = lambda value: top + h - value * h
    parts.extend([line(xmap(0), ymap(0), xmap(1), ymap(1), "4 4", 1),
                  text(left - 8, top + h + 4, "0", 10, "end"), text(left - 8, top + 4, "1", 10, "end"),
                  text(left, top + h + 16, "0", 10, "middle"), text(left + w, top + h + 16, "1", 10, "middle"),
                  text(left - 35, top + h / 2, "accuracy", 10, "middle")])
    raw_points, calibrated_points = [], []
    for row in metrics["calibration"]["reliability"]:
        raw_points.append((xmap(float(row["raw_confidence"])), ymap(float(row["accuracy"]))))
        calibrated_points.append((xmap(float(row["calibrated_confidence"])), ymap(float(row["accuracy"]))))
    parts.extend([polyline(raw_points, width=2), polyline(calibrated_points, "6 3", 2)])
    for x, y in raw_points:
        parts.append(circle(x, y, 4))
    for x, y in calibrated_points:
        parts.append(square(x, y, 4))
    parts.extend([text(left + 8, top + 18, "○ raw", 10), text(left + 8, top + 34, "■ calibrated", 10)])

    # Panel C: conditional versus marginal risk along the same abstention sweep.
    left, top, w, h = 805, 55, 300, 235
    axes(parts, left, top, w, h, "C  Abstention changes two risks", "coverage (fraction answered)")
    xmap = lambda value: left + value * w
    ymap = lambda value: top + h - value / 0.35 * h
    parts.extend([text(left - 8, top + h + 4, "0", 10, "end"), text(left - 8, top + 4, "0.35", 10, "end"),
                  text(left, top + h + 16, "0", 10, "middle"), text(left + w, top + h + 16, "1", 10, "middle"),
                  text(left - 40, top + h / 2, "error rate", 10, "middle")])
    selective = [(xmap(float(row["coverage"])), ymap(float(row["selective_risk"]))) for row in metrics["risk_curve"]]
    marginal = [(xmap(float(row["coverage"])), ymap(float(row["marginal_error"]))) for row in metrics["risk_curve"]]
    parts.extend([polyline(selective, width=2), polyline(marginal, "6 3", 2)])
    for x, y in selective:
        parts.append(circle(x, y, 4))
    for x, y in marginal:
        parts.append(square(x, y, 4))
    crc = metrics["crc"]
    cx, cy = xmap(float(crc["coverage"])), ymap(float(crc["marginal_error"]))
    parts.extend([circle(cx, cy, 8), text(cx + 10, cy - 7, "CRC α=0.10", 10),
                  text(left + 8, top + 18, "○ selective risk", 10), text(left + 8, top + 34, "■ marginal error", 10)])

    parts.append(text(WIDTH / 2, HEIGHT - 9, "All values are deterministic analytical or synthetic fixtures; they are not measurements of a deployed model or accelerator.", 10, "middle"))
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("inference_metrics.svg")
    render(destination)
