"""A deterministic repository-map, patch, test, and repair teaching agent."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


FIXTURE = Path(__file__).with_name("fixtures") / "tasks.json"
MAX_OBSERVATION_CHARS = 1_200


class EditRejected(ValueError):
    """Raised when a proposed edit violates the application contract."""


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str
    reason: str


@dataclass
class TaskResult:
    task_id: str
    resolved: bool = False
    proposals: int = 0
    test_runs: int = 0
    rejected_edits: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


def load_tasks(path: Path = FIXTURE) -> list[dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("fixture must contain a non-empty task list")
    return tasks


def safe_path(root: Path, relative: str, *, allow_tests: bool = False) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise EditRejected(f"path escapes workspace: {relative}")
    if candidate.suffix != ".py":
        raise EditRejected(f"only Python source files are editable: {relative}")
    if not allow_tests and candidate.name.startswith("test_"):
        raise EditRejected(f"tests are read-only: {relative}")
    return candidate


def materialize(task: dict[str, Any], root: Path) -> None:
    for relative, contents in task["files"].items():
        target = safe_path(root, relative, allow_tests=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")


def repository_map(root: Path) -> list[dict[str, Any]]:
    """Return paths and top-level symbols, not complete file contents."""
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
            symbols = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
        except SyntaxError:
            symbols = ["<syntax-error>"]
        result.append({"path": relative, "lines": source.count("\n") + 1, "symbols": symbols})
    return result


def apply_edit(root: Path, edit: Edit) -> str:
    """Apply one exact replacement and return its unified-diff receipt."""
    target = safe_path(root, edit.path)
    if not target.is_file():
        raise EditRejected(f"file does not exist: {edit.path}")
    before = target.read_text(encoding="utf-8")
    occurrences = before.count(edit.old)
    if occurrences != 1:
        raise EditRejected(f"old text must occur once, found {occurrences}: {edit.path}")
    after = before.replace(edit.old, edit.new, 1)
    try:
        ast.parse(after)
    except SyntaxError as exc:
        raise EditRejected(f"edit creates invalid Python: {exc.msg}") from exc
    target.write_text(after, encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{edit.path}",
            tofile=f"b/{edit.path}",
        )
    )


def run_tests(root: Path, timeout_seconds: float = 4.0) -> tuple[bool, str]:
    """Run the fixture's authoritative final-state check."""
    command = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = (completed.stdout + completed.stderr)[-MAX_OBSERVATION_CHARS:]
        return completed.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_seconds:.1f}s"


def solve_task(task: dict[str, Any], max_proposals: int = 3) -> TaskResult:
    """Run a scripted proposal adapter through the real edit-and-test loop."""
    result = TaskResult(task_id=task["id"])
    with tempfile.TemporaryDirectory(prefix=f"ch21-{task['id']}-") as directory:
        root = Path(directory).resolve()
        materialize(task, root)
        result.events.append({"kind": "observe", "repo_map": repository_map(root)})

        for raw in task["proposals"][:max_proposals]:
            result.proposals += 1
            edit = Edit(**raw)
            result.events.append({"kind": "propose", "edit": asdict(edit)})
            try:
                receipt = apply_edit(root, edit)
            except EditRejected as exc:
                result.rejected_edits += 1
                result.events.append({"kind": "reject", "reason": str(exc)})
                continue

            result.events.append({"kind": "execute", "diff": receipt})
            passed, observation = run_tests(root)
            result.test_runs += 1
            result.events.append(
                {"kind": "verify", "passed": passed, "observation": observation}
            )
            if passed:
                result.resolved = True
                result.events.append({"kind": "commit", "repo_map": repository_map(root)})
                break
    return result


def run_suite(tasks: list[dict[str, Any]], max_proposals: int = 3) -> dict[str, Any]:
    results = [solve_task(task, max_proposals=max_proposals) for task in tasks]
    return {
        "fixture": "deterministic-scripted-proposals",
        "tasks": len(results),
        "resolved": sum(result.resolved for result in results),
        "proposals": sum(result.proposals for result in results),
        "test_runs": sum(result.test_runs for result in results),
        "rejected_edits": sum(result.rejected_edits for result in results),
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_proposals < 1:
        parser.error("--max-proposals must be positive")

    report = run_suite(load_tasks(args.fixture), args.max_proposals)
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["resolved"] == report["tasks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
