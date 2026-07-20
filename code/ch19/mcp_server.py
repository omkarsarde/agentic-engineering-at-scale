"""Optional live MCP adapter for the dependency-free Chapter 19 domain service."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from protocol_lab import Principal, SERVER_AUDIENCE, SupportServer


service = SupportServer()
principal = Principal(
    subject="local-reader",
    tenant_id="tenant-7",
    audience=SERVER_AUDIENCE,
    scopes=frozenset({"policy:read", "case:write"}),
)
mcp = FastMCP("support-contracts", json_response=True)


@mcp.resource("policy://refund/current")
def current_refund_policy() -> str:
    """Read the current refund policy resource."""
    return service.read_resource("policy://refund/current", principal)["contents"][0]["text"]


@mcp.tool()
def policy_lookup(topic: str) -> dict:
    """Read the current refund policy for the authenticated tenant."""
    return service.call_tool("policy_lookup", {"topic": topic}, principal)["structuredContent"]


@mcp.tool()
def case_tag(case_id: str, tag: str, idempotency_key: str) -> dict:
    """Apply a reviewed routing tag to a support case."""
    return service.call_tool(
        "case_tag",
        {"case_id": case_id, "tag": tag, "idempotency_key": idempotency_key},
        principal,
    )["structuredContent"]


if __name__ == "__main__":
    mcp.run(transport="stdio")
