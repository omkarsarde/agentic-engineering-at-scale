"""Focused invariants for the Chapter 17 harness."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch17"))
sys.modules.pop("fixture", None)

from fixture import run_attack, run_resume, run_surface_eval  # noqa: E402
from state import Workspace  # noqa: E402


def test_substitution_and_stale_target_are_contained() -> None:
    outcomes = run_attack()
    assert "substituted action" in outcomes["substituted"]
    assert "stale target" in outcomes["stale"]
    assert outcomes["fresh"]["refunded_cents"] == 4_999


def test_resume_deduplicates_audit_event() -> None:
    result = run_resume()
    assert result["checkpoint"] == {"phase": "approved", "step": 2}
    assert result["duplicate_inserted"] is False
    assert result["audit_rows"] == 1


def test_deferred_tool_surface_preserves_recall() -> None:
    result = run_surface_eval()
    assert result["recall_at_2"] == 1.0
    assert result["retrieved_schema_chars"] < result["preloaded_schema_chars"]


def test_workspace_rejects_parent_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "thread-1")
    workspace.root.mkdir()
    assert workspace.resolve("notes/result.txt").is_relative_to(workspace.root)
    try:
        workspace.resolve("../other-thread/secret.txt")
    except PermissionError:
        pass
    else:
        raise AssertionError("workspace escape was not rejected")
