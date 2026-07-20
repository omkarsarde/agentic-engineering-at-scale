"""Render the Chapter 8 deterministic metrics as a print-safe SVG."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from reason_rl import run_experiment


STYLES = ["", "stroke-dasharray:9 5", "stroke-dasharray:2 5"]


def draw_panel(
    chunks: list[str], left: int, title: str, rows: list[dict[str, float]],
    x_key: str, series: list[tuple[str, str]], log_x: bool = False,
) -> None:
    width, top, bottom = 310, 60, 300
    xs = [math.log2(row[x_key]) if log_x else row[x_key] for row in rows]
    low, high = min(xs), max(xs)
    chunks += [f'<text class="title" x="{left}" y="28">{title}</text>',
               f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left+width}" y2="{bottom}"/>',
               f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
               f'<text x="{left-8}" y="304" text-anchor="end">0</text>',
               f'<text x="{left-8}" y="64" text-anchor="end">1</text>']
    for index, (key, label) in enumerate(series):
        points = []
        for x_value, row in zip(xs, rows):
            x = left + (x_value - low) / (high - low) * width
            y = bottom - row[key] * (bottom - top)
            points.append(f"{x:.1f},{y:.1f}")
        style = STYLES[index % len(STYLES)]
        chunks.append(f'<polyline class="series" style="{style}" points="{" ".join(points)}"/>')
        chunks.append(f'<text x="{left+8}" y="{338+index*17}">{index+1}. {label}</text>')
    for x_value, row in zip(xs, rows):
        x = left + (x_value - low) / (high - low) * width
        label = f'{row[x_key]:.0f}'
        chunks += [f'<line class="axis" x1="{x:.1f}" y1="300" x2="{x:.1f}" y2="305"/>',
                   f'<text x="{x:.1f}" y="320" text-anchor="middle">{label}</text>']


def write_svg(path: Path) -> None:
    """Generate one three-panel figure from the reference experiment."""
    result = run_experiment()
    exact = result["training"]["exact"]
    proxy = result["training"]["proxy"]
    combined = [dict(row, proxy_accuracy=other["true_accuracy"],
                     proxy_objective=other["normalized_objective"],
                     proxy_entropy=other["normalized_entropy"])
                for row, other in zip(exact, proxy)]
    chunks = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="405" viewBox="0 0 1200 405">',
              '<rect width="1200" height="405" fill="white"/>',
              '<style>text{font:13px sans-serif;fill:#111}.title{font:bold 15px sans-serif}.axis{stroke:#111;stroke-width:1}.series{fill:none;stroke:#111;stroke-width:3}</style>']
    draw_panel(chunks, 55, "Inference accuracy vs generated tokens", result["ttc"],
               "mean_tokens", [("coverage", "exact-verifier coverage"),
                               ("plurality", "plurality"),
                               ("noisy_bon", "noisy verifier")], log_x=True)
    draw_panel(chunks, 455, "GRPO objective vs true accuracy", combined, "step",
               [("true_accuracy", "exact-RLVR accuracy"),
                ("proxy_accuracy", "proxy-run accuracy"),
                ("proxy_objective", "proxy objective / maximum")])
    draw_panel(chunks, 855, "Exploration during GRPO", combined, "step",
               [("normalized_entropy", "exact-RLVR entropy"),
                ("proxy_entropy", "proxy-run entropy")])
    chunks += ['<text x="175" y="398">Generated tokens (log2 spacing; labels show tokens)</text>',
               '<text x="590" y="398">Training step</text>',
               '<text x="990" y="398">Training step</text>', '</svg>']
    path.write_text("\n".join(chunks), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("reasoning_metrics.svg"))
    write_svg(parser.parse_args().output)
