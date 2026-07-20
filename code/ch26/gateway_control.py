"""Identity, admission, routing, and retry controls for the mini platform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RequestContext:
    """Identity derived from a virtual key, never from model-controlled fields."""

    request_id: str
    tenant: str
    scopes: frozenset[str]
    deadline_s: float


@dataclass(frozen=True)
class Usage:
    """Metered tokens and the price schedule used to settle them."""

    input_tokens: int
    output_tokens: int
    input_per_million: float
    output_per_million: float

    @property
    def cost(self) -> float:
        return (
            self.input_tokens * self.input_per_million
            + self.output_tokens * self.output_per_million
        ) / 1_000_000


@dataclass
class Deployment:
    name: str
    capabilities: frozenset[str]
    cost_per_million: float
    healthy: bool = True


class BudgetError(RuntimeError):
    """Raised when an entire journey has no remaining spend authority."""


class Gateway:
    """Authenticate, admit, route, and attribute an agent request."""

    def __init__(
        self,
        keys: dict[str, tuple[str, frozenset[str]]],
        limits: dict[str, float],
        deployments: list[Deployment],
    ) -> None:
        self.keys = keys
        self.limits = limits
        self.deployments = deployments
        self.spend = {tenant: 0.0 for tenant in limits}

    def authenticate(
        self, virtual_key: str, request_id: str, deadline_s: float
    ) -> RequestContext:
        if virtual_key not in self.keys:
            raise PermissionError("unknown virtual key")
        tenant, scopes = self.keys[virtual_key]
        return RequestContext(request_id, tenant, scopes, deadline_s)

    def admit(self, context: RequestContext, estimated_cost: float) -> None:
        limit = self.limits[context.tenant]
        if self.spend[context.tenant] + estimated_cost > limit:
            raise BudgetError("tenant journey budget exhausted")

    def route(self, context: RequestContext, capability: str) -> Deployment:
        candidates = [
            item
            for item in self.deployments
            if capability in item.capabilities and item.healthy
        ]
        if not candidates:
            raise LookupError("no healthy deployment satisfies the capability")
        return min(candidates, key=lambda item: item.cost_per_million)

    def settle(self, context: RequestContext, usage: Usage) -> float:
        self.spend[context.tenant] += usage.cost
        return self.spend[context.tenant]


TERMINAL_ERRORS = frozenset({"policy_refusal", "invalid_request", "auth"})
RETRYABLE_ERRORS = frozenset({"overloaded", "connection_reset", "rate_limited"})


def fallback_allowed(error_kind: str, remaining_s: float) -> bool:
    """Permit provider fallback only for transient failures with time remaining."""
    return error_kind in RETRYABLE_ERRORS and remaining_s > 0


@dataclass
class RetryBudget:
    attempts_left: int
    deadline_s: float
    now: Callable[[], float]

    def consume(self, error_kind: str) -> float:
        remaining = self.deadline_s - self.now()
        if error_kind in TERMINAL_ERRORS:
            raise RuntimeError("terminal failure")
        if error_kind not in RETRYABLE_ERRORS or self.attempts_left <= 0 or remaining <= 0:
            raise TimeoutError("retry budget exhausted")
        self.attempts_left -= 1
        return remaining
