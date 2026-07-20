"""Deterministic document extraction and evaluation fixture for Chapter 29."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


FIELDS = ("invoice_id", "vendor", "total", "due_date", "chart_peak")


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Page:
    page_id: str
    width: int
    height: int
    gold: dict[str, str]
    regions: dict[str, Box]


def make_pages(count: int = 12) -> list[Page]:
    """Create source-traceable invoice pages with field and region truth."""
    pages: list[Page] = []
    vendors = ("Northwind", "Contoso", "Tailspin")
    for index in range(count):
        gold = {
            "invoice_id": f"INV-{1000 + index}",
            "vendor": vendors[index % len(vendors)],
            "total": f"{125 + 17 * index}.00",
            "due_date": f"2026-08-{10 + index:02d}",
            "chart_peak": ("Q4", "Q3", "Q2")[index % 3],
        }
        pages.append(
            Page(
                f"page-{index:02d}",
                1200,
                1600,
                gold,
                {
                    "invoice_id": Box(80, 80, 360, 140),
                    "vendor": Box(80, 180, 480, 260),
                    "total": Box(820, 1180, 1080, 1260),
                    "due_date": Box(780, 220, 1080, 290),
                    "chart_peak": Box(180, 650, 920, 1040),
                },
            )
        )
    return pages


def image_tokens(tiles: int, tokens_per_tile: int = 256, thumbnail: int = 64) -> int:
    """Expose the illustrative linear token bill for dynamic tiling."""
    if tiles <= 0:
        raise ValueError("tiles must be positive")
    return tiles * tokens_per_tile + thumbnail


def fixture_extract(page: Page, tiles: int) -> dict[str, dict[str, Any]]:
    """Emulate resolution-dependent extraction while retaining evidence boxes."""
    values = dict(page.gold)
    index = int(page.page_id.rsplit("-", 1)[1])
    if tiles == 1:
        values["chart_peak"] = "unknown"
        if index % 3 == 0:
            values["due_date"] = "unknown"
    elif tiles == 4:
        if index % 4 == 0:
            values["chart_peak"] = "unknown"
    elif tiles >= 9:
        if index == 11:
            values["chart_peak"] = "Q1"
    return {
        field: {"value": values[field], "box": page.regions[field].__dict__}
        for field in FIELDS
    }


def validate_extraction(extraction: dict[str, dict[str, Any]], page: Page) -> None:
    """Reject missing fields, wrong value types, and out-of-page evidence."""
    if set(extraction) != set(FIELDS):
        raise ValueError("extraction does not match the field schema")
    for field, item in extraction.items():
        if not isinstance(item.get("value"), str):
            raise TypeError(f"{field} value must be a string")
        box = item.get("box", {})
        required = {"x0", "y0", "x1", "y1"}
        if set(box) != required:
            raise ValueError(f"{field} lacks an evidence box")
        if not (0 <= box["x0"] < box["x1"] <= page.width):
            raise ValueError(f"{field} evidence exceeds page width")
        if not (0 <= box["y0"] < box["y1"] <= page.height):
            raise ValueError(f"{field} evidence exceeds page height")


def score_fields(
    extraction: dict[str, dict[str, Any]], page: Page
) -> tuple[int, int]:
    """Count exact normalized field matches against authoritative page truth."""
    correct = sum(
        extraction[field]["value"].strip().casefold()
        == page.gold[field].strip().casefold()
        for field in FIELDS
    )
    return correct, len(FIELDS)


class OpenAICompatibleVLM:
    """Optional adapter for a local OpenAI-compatible multimodal endpoint."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def extract(self, image_path: Path) -> dict[str, dict[str, Any]]:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        prompt = (
            "Extract invoice_id, vendor, total, due_date, and chart_peak. "
            "For every field return {value, box:{x0,y0,x1,y1}}. Return JSON only."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encoded}"
                }},
            ]}],
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])


def evaluate(tiles: int, pages: list[Page] | None = None) -> dict[str, float | int]:
    pages = pages or make_pages()
    correct = total = 0
    for page in pages:
        extraction = fixture_extract(page, tiles)
        validate_extraction(extraction, page)
        page_correct, page_total = score_fields(extraction, page)
        correct += page_correct
        total += page_total
    return {
        "tiles": tiles,
        "image_tokens_per_page": image_tokens(tiles),
        "correct_fields": correct,
        "total_fields": total,
        "field_accuracy": correct / total,
    }


def run_sweep() -> list[dict[str, float | int]]:
    return [evaluate(tiles) for tiles in (1, 4, 9, 16)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.base_url and args.model and args.image:
        print(json.dumps(OpenAICompatibleVLM(args.base_url, args.model).extract(args.image), indent=2))
    else:
        print(json.dumps(run_sweep(), indent=2))
