"""WheelBot Ops MCP entrypoint.

    docker exec / compose service: python -m scripts.run_mcp

Serves the read tools + guarded controls over authenticated streamable HTTP.
The bearer token is the primary guard (controls ship ON), so the server
REFUSES to start without a strong `WHEELBOT_MCP_TOKEN`.

Config (config.yaml `mcp:` block):
    mcp:
      host: "0.0.0.0"          # bound to a private interface by the compose port map
      port: 8890
      controls_enabled: true

Setup is fully synchronous: uvicorn owns the event loop, and the aiosqlite
connection is created lazily on the first tool call (inside uvicorn's loop) so
it isn't bound to a throwaway setup loop.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from core.broker_factory import make_broker
from core.checkpoint import configure_logging, log_checkpoint
from core.config import load_config
from core.logs import setup_logging
from db.repo import Database, Repos
from mcp_server.server import build_server, run_http
from mcp_server.service import WheelbotMcpService

_MIN_TOKEN_LEN = 16


def main() -> int:
    configure_logging()
    config = load_config()
    setup_logging(config)

    token = os.environ.get("WHEELBOT_MCP_TOKEN", "")
    if len(token) < _MIN_TOKEN_LEN:
        log_checkpoint(
            "mcp_start", status="fail",
            reason=f"WHEELBOT_MCP_TOKEN missing or < {_MIN_TOKEN_LEN} chars",
        )
        print(
            f"ERROR: set WHEELBOT_MCP_TOKEN (>= {_MIN_TOKEN_LEN} chars) before starting the MCP.",
            file=sys.stderr,
        )
        return 2

    mcp_cfg = config.get("mcp", {})
    host = mcp_cfg.get("host", "0.0.0.0")
    port = int(mcp_cfg.get("port", 8890))
    controls_enabled = bool(mcp_cfg.get("controls_enabled", True))

    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    db = Database(db_path)  # not connected — repos connect lazily in uvicorn's loop
    repos = Repos(db)
    broker = make_broker(config)
    service = WheelbotMcpService(repos, broker, config, controls_enabled=controls_enabled)
    server = build_server(service, host=host, port=port)

    log_checkpoint(
        "mcp_start", status="ok", host=host, port=port, controls_enabled=controls_enabled,
    )
    run_http(server, token=token)  # blocks (uvicorn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
