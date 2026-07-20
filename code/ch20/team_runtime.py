"""A cost-accounted orchestrator-worker runtime for Chapter 20."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskBrief:
    """Authority and output contract delegated to one worker."""

    worker_id: str
    shard_id: str
    allowed_tools: tuple[str, ...]
    output_schema: str
    token_budget: int
    deadline_units: int


@dataclass(frozen=True)
class WorkerReport:
    """A typed handoff artifact with evidence rather than a chat summary."""

    worker_id: str
    shard_id: str
    finding: str
    evidence_ids: tuple[str, ...]
    tokens: int
    latency_units: int


@dataclass(frozen=True)
class Failure:
    """A coarse MAST-style failure attribution."""

    category: str
    agent: str
    step: str
    reason: str


@dataclass(frozen=True)
class RunResult:
    """Comparable quality, budget, latency, and attribution for one run."""

    architecture: str
    status: str
    answer: str | None
    score: float
    tokens: int
    latency_units: int
    failure: Failure | None = None


class Workspace:
    """Store immutable worker reports under a parent-owned directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, report: WorkerReport) -> Path:
        path = (self.root / f"{report.worker_id}.json").resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError("report path escapes workspace")
        if path.exists():
            raise FileExistsError(f"report already exists: {path.name}")
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return path

    def read(self, path: Path) -> WorkerReport:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("report path escapes workspace")
        data = json.loads(resolved.read_text(encoding="utf-8"))
        data["evidence_ids"] = tuple(data["evidence_ids"])
        return WorkerReport(**data)


class Worker:
    """Read exactly one shard and produce one bounded report."""

    def run(
        self,
        brief: TaskBrief,
        shard: dict[str, str],
        workspace: Workspace,
        poison: bool = False,
    ) -> Path:
        if brief.allowed_tools != ("read_shard",):
            raise PermissionError("worker received unexpected authority")
        if shard["shard_id"] != brief.shard_id:
            raise ValueError("brief does not match shard")
        evidence = () if poison else (shard["evidence_id"],)
        report = WorkerReport(
            brief.worker_id,
            brief.shard_id,
            shard["finding"],
            evidence,
            tokens=220,
            latency_units=5,
        )
        return workspace.write(report)


def _validate_report(report: WorkerReport, shard: dict[str, str]) -> Failure | None:
    if report.shard_id != shard["shard_id"]:
        return Failure("inter-agent misalignment", report.worker_id, "handoff", "wrong shard")
    if report.finding != shard["finding"]:
        return Failure("inter-agent misalignment", report.worker_id, "handoff", "unsupported finding")
    if report.evidence_ids != (shard["evidence_id"],):
        return Failure("inter-agent misalignment", report.worker_id, "handoff", "missing provenance")
    return None


class Orchestrator:
    """Fan out read-only work and remain the only final-state writer."""

    PLAN_TOKENS = 180
    MERGE_TOKENS = 300

    def run(
        self,
        corpus: tuple[dict[str, str], ...],
        workspace: Workspace,
        poison_worker: str | None = None,
    ) -> RunResult:
        briefs = tuple(
            TaskBrief(
                f"worker-{index}",
                shard["shard_id"],
                ("read_shard",),
                "WorkerReport/v1",
                220,
                8,
            )
            for index, shard in enumerate(corpus, start=1)
        )
        with ThreadPoolExecutor(max_workers=len(briefs)) as pool:
            futures = [
                pool.submit(
                    Worker().run,
                    brief,
                    shard,
                    workspace,
                    brief.worker_id == poison_worker,
                )
                for brief, shard in zip(briefs, corpus, strict=True)
            ]
            paths = [future.result() for future in futures]

        reports = [workspace.read(path) for path in paths]
        tokens = self.PLAN_TOKENS + sum(report.tokens for report in reports) + self.MERGE_TOKENS
        latency = 2 + max(report.latency_units for report in reports) + 3
        for report, shard in zip(reports, corpus, strict=True):
            if failure := _validate_report(report, shard):
                return RunResult("orchestrator-worker", "contained", None, 0.0, tokens, latency, failure)
        answer = ", ".join(sorted(report.finding for report in reports))
        return RunResult("orchestrator-worker", "completed", answer, 1.0, tokens, latency)


class SingleAgent:
    """Solve the same fixture sequentially at the measured team budget."""

    def run(self, corpus: tuple[dict[str, str], ...], token_budget: int) -> RunResult:
        minimum = 180 + 220 * len(corpus) + 300
        if token_budget < minimum:
            return RunResult("single-agent", "budget_exhausted", None, 0.0, token_budget, 0)
        answer = ", ".join(sorted(shard["finding"] for shard in corpus))
        latency = 2 + 5 * len(corpus) + 3
        return RunResult("single-agent", "completed", answer, 1.0, minimum, latency)


def should_split(*, parallelizable: bool, isolation_boundary: bool, measured_failure: bool) -> bool:
    """Require all three positive conditions before adding agent coordination."""
    return parallelizable and isolation_boundary and measured_failure


def result_dict(result: RunResult) -> dict[str, Any]:
    data = asdict(result)
    if data["failure"] is None:
        data.pop("failure")
    return data
