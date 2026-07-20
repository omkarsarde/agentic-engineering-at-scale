"""Integrated public API and deterministic build for Chapter 13.

The experiment stays one coherent pipeline while its prompt search, typed
assembly, and history measurement live in focused sibling modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_assembly import (
    ORDER,
    Segment,
    compress_context,
    isolate_context,
    render_context,
    select_context,
    token_count,
    write_context,
)
from context_history import (
    DECISIONS,
    OPEN_QUESTIONS,
    assert_survival,
    cache_cost_example,
    compact_history,
    common_prefix_bytes,
    simulate_turns,
    synthetic_history,
)
from prompt_optimizer import (
    BASELINE_PROMPT,
    CANDIDATE_PROMPTS,
    GOLDEN_SET,
    Ticket,
    classify,
    evaluate,
    optimize_prompt,
)


def run_build() -> dict:
    """Run prompt search, compaction checks, and the prefix experiment."""
    optimization = optimize_prompt()
    history = synthetic_history()
    compacted, changed = compact_history(history, budget=500)
    assert_survival(history, compacted)
    simulations = [
        *simulate_turns(False, True),
        *simulate_turns(False, False),
        *simulate_turns(True, True),
        *simulate_turns(True, False),
    ]
    return {
        "prompt_optimization": optimization,
        "compaction": {
            "before_messages": len(history),
            "after_messages": len(compacted),
            "before_tokens": token_count("\n".join(history)),
            "after_tokens": token_count("\n".join(compacted)),
            "changed": changed,
            "lost_decisions": 0,
            "lost_open_questions": 0,
        },
        "cache_cost": cache_cost_example(),
        "ledger": simulations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_build()
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
