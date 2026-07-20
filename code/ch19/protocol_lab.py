"""A dependency-free protocol seam used by the Chapter 19 build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


PROTOCOL_VERSION = "2025-11-25"
SERVER_AUDIENCE = "support://mcp"


class ProtocolError(RuntimeError):
    """A typed protocol, validation, or authorization failure."""


@dataclass(frozen=True)
class Principal:
    """Identity established by the transport, never by model arguments."""

    subject: str
    tenant_id: str
    audience: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class Tool:
    """A discoverable operation with one required authorization scope."""

    name: str
    description: str
    schema: dict[str, Any]
    scope: str
    handler: Callable[[dict[str, Any], Principal], dict[str, Any]]


def _require_keys(arguments: dict[str, Any], required: set[str]) -> None:
    missing = required - arguments.keys()
    extra = arguments.keys() - required
    if missing or extra:
        raise ProtocolError(f"invalid arguments: missing={sorted(missing)}, extra={sorted(extra)}")


class SupportServer:
    """Own protocol negotiation, discovery, authorization, and domain effects."""

    def __init__(self) -> None:
        self.case_tags: dict[str, str] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.tools = {
            "policy_lookup": Tool(
                "policy_lookup",
                "Read the current refund policy for an authenticated tenant.",
                {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
                "policy:read",
                self._policy_lookup,
            ),
            "case_tag": Tool(
                "case_tag",
                "Apply a reviewed routing tag to a support case.",
                {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "tag": {"enum": ["manual_review", "eligible"]},
                        "idempotency_key": {"type": "string"},
                    },
                    "required": ["case_id", "tag", "idempotency_key"],
                },
                "case:write",
                self._case_tag,
            ),
        }

    @staticmethod
    def _authorize(principal: Principal, required_scope: str) -> None:
        if principal.audience != SERVER_AUDIENCE:
            raise ProtocolError("invalid token audience")
        if required_scope not in principal.scopes:
            raise ProtocolError(f"missing scope: {required_scope}")

    def initialize(self, requested: str) -> dict[str, Any]:
        """Negotiate an explicit protocol revision and server capabilities."""
        if requested != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {requested}")
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": "support-contracts", "version": "1.0.0"},
            "capabilities": {"tools": {}, "resources": {}},
        }

    def list_tools(self, principal: Principal) -> dict[str, Any]:
        """Expose only tools the authenticated principal may call."""
        self._authorize(principal, "policy:read")
        visible = [tool for tool in self.tools.values() if tool.scope in principal.scopes]
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema,
                }
                for tool in sorted(visible, key=lambda item: item.name)
            ]
        }

    def read_resource(self, uri: str, principal: Principal) -> dict[str, Any]:
        """Return application-controlled context through a stable URI."""
        self._authorize(principal, "policy:read")
        if uri != "policy://refund/current":
            raise ProtocolError(f"unknown resource: {uri}")
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": '{"version":"v3","manual_review_above_cents":5000}',
                }
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        """Authorize and execute a discovered operation."""
        tool = self.tools.get(name)
        if tool is None:
            raise ProtocolError(f"unknown tool: {name}")
        self._authorize(principal, tool.scope)
        result = tool.handler(arguments, principal)
        return {"content": [{"type": "text", "text": str(result)}], "structuredContent": result}

    def dispatch(self, request: dict[str, Any], principal: Principal) -> dict[str, Any]:
        """Handle the small JSON-RPC-shaped subset used by the fixture."""
        if request.get("jsonrpc") != "2.0" or "id" not in request:
            raise ProtocolError("request must be correlated JSON-RPC 2.0")
        method, params = request.get("method"), request.get("params", {})
        if method == "initialize":
            result = self.initialize(params["protocolVersion"])
        elif method == "tools/list":
            result = self.list_tools(principal)
        elif method == "resources/read":
            result = self.read_resource(params["uri"], principal)
        elif method == "tools/call":
            result = self.call_tool(params["name"], params["arguments"], principal)
        else:
            raise ProtocolError(f"unsupported method: {method}")
        return {"jsonrpc": "2.0", "id": request["id"], "result": result}

    @staticmethod
    def _policy_lookup(arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        _require_keys(arguments, {"topic"})
        if arguments["topic"] != "refund":
            return {"found": False, "tenant_id": principal.tenant_id}
        return {
            "found": True,
            "tenant_id": principal.tenant_id,
            "policy_version": "v3",
            "manual_review_above_cents": 5000,
        }

    def _case_tag(self, arguments: dict[str, Any], principal: Principal) -> dict[str, Any]:
        _require_keys(arguments, {"case_id", "tag", "idempotency_key"})
        key = arguments["idempotency_key"]
        if key in self.receipts:
            return self.receipts[key]
        case_key = f"{principal.tenant_id}:{arguments['case_id']}"
        self.case_tags[case_key] = arguments["tag"]
        receipt = {"case_id": arguments["case_id"], "tag": arguments["tag"], "receipt": key}
        self.receipts[key] = receipt
        return receipt
