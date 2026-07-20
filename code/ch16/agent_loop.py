"""A bounded proposal-gate-execute-observe kernel for Chapter 16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable, Protocol


class Stop(StrEnum):
    """Machine-readable ways a run can end."""

    ANSWERED = "answered"
    DENIED = "denied"
    NO_PROGRESS = "no_progress"
    STEP_LIMIT = "step_limit"
    COST_LIMIT = "cost_limit"


@dataclass(frozen=True)
class ToolCall:
    """A model proposal; it has no authority by itself."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    """A model proposal to stop and answer the caller."""

    text: str


Decision = ToolCall | FinalAnswer


@dataclass(frozen=True)
class Observation:
    """A correlated, typed result returned to the strategy."""

    call_id: str
    ok: bool
    kind: str
    content: Any


@dataclass(frozen=True)
class ToolSpec:
    """Bind a name to its argument contract and local effect handler."""

    schema: dict[str, type]
    handler: Callable[..., Any]
    effectful: bool = False


@dataclass(frozen=True)
class Limits:
    """Bound resources owned by this in-process teaching kernel."""

    max_steps: int = 8
    max_cost_units: int = 16
    repeated_denials: int = 2


@dataclass
class RunState:
    """Serializable state visible to a strategy at each boundary."""

    task: str
    observations: list[Observation] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    cost_units: int = 0


@dataclass(frozen=True)
class RunResult:
    """The typed outcome consumed by a caller or evaluator."""

    stop: Stop
    answer: str | None
    state: RunState


class Strategy(Protocol):
    """Choose the next proposal from explicit run state."""

    def decide(self, state: RunState) -> Decision: ...


Gate = Callable[[ToolCall, RunState], tuple[bool, str]]


def _validate(call: ToolCall, tool: ToolSpec) -> str | None:
    """Return a validation error, or None when arguments match exactly."""
    if set(call.arguments) != set(tool.schema):
        return f"expected fields {sorted(tool.schema)}"
    for name, expected in tool.schema.items():
        if not isinstance(call.arguments[name], expected):
            return f"{name} must be {expected.__name__}"
    return None


def _fingerprint(call: ToolCall) -> str:
    """Create a stable identity for no-progress detection."""
    payload = json.dumps(
        {"name": call.name, "arguments": call.arguments}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_agent(
    task: str,
    strategy: Strategy,
    tools: dict[str, ToolSpec],
    gate: Gate,
    limits: Limits = Limits(),
) -> RunResult:
    """Run one bounded task through proposal, gate, execution, and observation.

    Args:
        task: Caller goal; it grants no tool authority.
        strategy: Model or deterministic policy that proposes the next decision.
        tools: Code-owned action registry.
        gate: Policy function evaluated immediately before every handler call.
        limits: Resource ceilings enforced by the kernel.

    Returns:
        A typed stop reason, optional answer, and complete in-process trace.
    """
    state = RunState(task=task)
    denied_counts: dict[str, int] = {}

    while state.steps < limits.max_steps:
        if state.cost_units + 2 > limits.max_cost_units:
            return RunResult(Stop.COST_LIMIT, None, state)

        decision = strategy.decide(state)
        state.steps += 1
        state.cost_units += 2
        state.trace.append({"event": "proposal", "value": repr(decision)})

        if isinstance(decision, FinalAnswer):
            return RunResult(Stop.ANSWERED, decision.text, state)

        tool = tools.get(decision.name)
        error = "unknown tool" if tool is None else _validate(decision, tool)
        allowed, reason = gate(decision, state) if error is None else (False, error)

        if not allowed:
            observation = Observation(decision.call_id, False, "denied", reason)
            state.observations.append(observation)
            state.trace.append({"event": "gate", "allowed": False, "reason": reason})
            fingerprint = _fingerprint(decision)
            denied_counts[fingerprint] = denied_counts.get(fingerprint, 0) + 1
            if denied_counts[fingerprint] >= limits.repeated_denials:
                return RunResult(Stop.NO_PROGRESS, None, state)
            continue

        try:
            content = tool.handler(**decision.arguments)
            observation = Observation(decision.call_id, True, "result", content)
        except TimeoutError as exc:
            observation = Observation(decision.call_id, False, "transient", str(exc))
        except (KeyError, ValueError) as exc:
            observation = Observation(decision.call_id, False, "permanent", str(exc))

        state.observations.append(observation)
        state.trace.append(
            {"event": "observation", "value": asdict(observation)}
        )

    return RunResult(Stop.STEP_LIMIT, None, state)
