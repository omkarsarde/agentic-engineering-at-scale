"""A small harness boundary for tools, approvals, workspaces, and resume state."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class Risk(StrEnum):
    """Risk class used to decide whether execution needs approval."""

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class Tool:
    """A model-facing contract bound to an application-owned handler."""

    name: str
    summary: str
    schema: dict[str, type]
    risk: Risk
    handler: Callable[..., Any]
    target_version: Callable[[dict[str, Any]], str] | None = None


@dataclass(frozen=True)
class Call:
    """An exact proposed tool invocation."""

    call_id: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ExecutionContext:
    """Authenticated context supplied by the application, never the model."""

    tenant_id: str
    actor_id: str


@dataclass(frozen=True)
class ApprovalRequest:
    """A reviewable proposal bound to action, target state, and expiry."""

    request_id: str
    action_digest: str
    target_digest: str
    expires_at: float


@dataclass(frozen=True)
class Approval:
    """A human decision over one immutable request."""

    request_id: str
    action_digest: str
    target_digest: str
    expires_at: float
    approver_id: str


class ApprovalError(RuntimeError):
    """The supplied approval does not authorize the current action."""


def _canonical(value: Any) -> bytes:
    """Serialize security-bound values deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    """Hash a canonical value for binding and comparison."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def action_digest(call: Call, context: ExecutionContext) -> str:
    """Bind a proposal to exact arguments, tenant, and authenticated actor."""
    return _digest(
        {
            "tool": call.tool,
            "arguments": call.arguments,
            "tenant_id": context.tenant_id,
            "actor_id": context.actor_id,
        }
    )


def request_approval(
    call: Call,
    context: ExecutionContext,
    target_version: str,
    ttl_s: float = 300,
    now: float | None = None,
) -> ApprovalRequest:
    """Create an immutable approval request for the current target state."""
    issued_at = time.time() if now is None else now
    action = action_digest(call, context)
    target = _digest({"target_version": target_version})
    return ApprovalRequest(action[:16], action, target, issued_at + ttl_s)


def approve(request: ApprovalRequest, approver_id: str) -> Approval:
    """Record a person's decision without changing the approved payload."""
    return Approval(
        request.request_id,
        request.action_digest,
        request.target_digest,
        request.expires_at,
        approver_id,
    )


def validate_call(call: Call, tool: Tool) -> None:
    """Reject unknown, missing, extra, or wrongly typed arguments."""
    if set(call.arguments) != set(tool.schema):
        raise ValueError(f"expected fields {sorted(tool.schema)}")
    for name, expected in tool.schema.items():
        if not isinstance(call.arguments[name], expected):
            raise ValueError(f"{name} must be {expected.__name__}")


def dispatch(
    call: Call,
    context: ExecutionContext,
    tools: dict[str, Tool],
    approval: Approval | None = None,
    now: float | None = None,
) -> Any:
    """Revalidate an exact action at the last responsible moment.

    Raises:
        ApprovalError: If a write lacks a fresh approval bound to the same
            action and target version.
        ValueError: If the tool contract is malformed.
    """
    tool = tools[call.tool]
    validate_call(call, tool)

    if tool.risk is Risk.WRITE:
        if approval is None:
            raise ApprovalError("write requires approval")
        current_time = time.time() if now is None else now
        current_target = tool.target_version(call.arguments) if tool.target_version else ""
        checks = {
            "expired": current_time > approval.expires_at,
            "substituted action": action_digest(call, context) != approval.action_digest,
            "stale target": _digest({"target_version": current_target})
            != approval.target_digest,
        }
        failed = [name for name, is_failed in checks.items() if is_failed]
        if failed:
            raise ApprovalError(", ".join(failed))

    return tool.handler(**call.arguments)


def select_tools(query: str, tools: list[Tool], limit: int = 3) -> list[Tool]:
    """Retrieve a small tool surface by transparent lexical overlap."""
    terms = set(query.casefold().replace("-", " ").split())

    def score(tool: Tool) -> tuple[int, str]:
        words = set(f"{tool.name} {tool.summary}".casefold().replace("_", " ").split())
        return len(terms & words), tool.name

    return sorted(tools, key=score, reverse=True)[:limit]
