# WheelBot Ops MCP

Operate and interrogate the running bot conversationally from Claude Code. A
small [MCP](https://modelcontextprotocol.io) server (`scripts/run_mcp`, the
`wheelbot-mcp` compose service) exposes read tools plus a few guarded controls
over authenticated HTTP. It reuses the bot's own DB + broker + control paths —
no new order logic.

## Tools

**Read (always available):**

| Tool | Answers |
|------|---------|
| `get_positions` | open positions: state, strategy, cost basis, cycle |
| `get_account_risk` | equity / cash / buying-power, net position mark, concurrent-cap usage |
| `get_strategy_status` | per-strategy config-enabled flag, runtime drawdown/pause state, open count |
| `get_performance` | realized P&L + win rate by strategy, recent closed cycles |
| `get_recent_decisions` | recent screener / news_check LLM decisions + today's LLM cost |
| `get_regime_and_calendar` | regime flags + upcoming macro events + calendar freshness |
| `diagnose_symbol` | why a symbol may not be trading: owning strategies, regime gate, cap, next earnings, recent orders |

**Guarded controls (require `mcp.controls_enabled: true`, shipped ON):**

| Tool | Guard |
|------|-------|
| `pause_strategy` / `reenable_strategy` | runtime gating only — places **no orders** |
| `engage_kill_switch` / `release_kill_switch` | global stop via the shared-volume stop file |
| `refresh_macro_calendar` | idempotent |
| `flatten_position(symbol, execute=false)` | **dry-run by default** — returns the plan; acts only with `execute=true` |

Every control writes an `mcp_control` checkpoint (`triggered_by=MCP`) for the
audit trail.

## Safety model

- **Bearer token is the primary guard.** The server **refuses to start** without
  `WHEELBOT_MCP_TOKEN` (≥ 16 chars), and rejects any request without
  `Authorization: Bearer <token>`.
- **Private binding.** The port is published to `127.0.0.1` on the LXC (like the
  dashboard); reach it over your tailnet (below), never the public internet.
- **`flatten_position` is dry-run-first** and only acts with `execute=true`.
- **`mcp.controls_enabled`** (config.yaml) can disable all control tools at once;
  it ships `true`. Read tools are unaffected.

## Setup

1. **Token** — add a strong secret to `config/secrets.env` on the LXC:
   ```
   WHEELBOT_MCP_TOKEN=<openssl rand -hex 24>
   ```
2. **Deploy** — rebuilds the image and starts the new `mcp` service alongside the
   bot + dashboard:
   ```
   cd /opt/wheelbot && git pull && docker compose up -d --build
   docker logs wheelbot-mcp --tail 20   # expect: mcp_start status=ok
   ```
3. **Expose to your tailnet** — the server listens on `127.0.0.1:8890` inside the
   LXC. Publish it to the tailnet the same way the dashboard (`:8889`) is, e.g.:
   ```
   tailscale serve --bg --https=8890 http://127.0.0.1:8890
   ```
   (or an SSH `-L 8890:127.0.0.1:8890` tunnel). This gives a stable, TLS,
   tailnet-only URL.

## Wire into Claude Code

From your dev box (on the tailnet):

```
claude mcp add --transport http wheelbot https://<lxc-tailnet-host>:8890/mcp \
  --header "Authorization: Bearer $WHEELBOT_MCP_TOKEN"
```

Then just ask, e.g. *"what's my open risk?"*, *"why isn't it trading QQQ?"*,
*"win rate by strategy"*, *"pause narrow_put_spread"*, *"dry-run flatten META"*.

## Notes

- The kill-switch stop file lives on the shared volume (`/mnt/wheelbot-storage/STOP`)
  so the MCP, bot, and dashboard containers all see it. (Before this, a duplicate
  `risk:` key in config.yaml had dropped `stop_file_path` entirely — the manual
  stop switch was a no-op.)
- `flatten_position` currently supports the Alpaca broker path; it returns a clear
  error on other brokers (Tastytrade flatten is a follow-up).
