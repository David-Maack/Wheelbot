"""WheelBot Ops MCP — operate and interrogate the bot conversationally.

`service.py` holds the read tools + guarded controls as a plain class (unit
tested directly). `server.py` is a thin FastMCP wrapper that exposes them as
MCP tools over authenticated HTTP. `scripts/run_mcp.py` is the entrypoint.
"""
