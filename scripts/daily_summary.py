"""Daily P&L + position summary, posted to Discord (Sprint 13 sub-sprint 3).

Designed to be invoked once per trading day via the LXC host crontab:

    30 16 * * 1-5 docker exec wheelbot python -m scripts.daily_summary

Pulls realized P&L from `wheel_cycles`, open positions from `positions`,
regime flags from `regime_snapshots`, and auto-disabled strategies from
`strategy_runtime_state`. Formats as a Discord embed via core.notify.

Read-only. Safe to run any time; if no metrics are available the embed
still posts (acts as a heartbeat too).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from core.config import load_config
from core.notify import make_notifier, notify, set_dispatcher
from db.repo import Database, Repos


async def compute_metrics(
    repos: Repos,
    config: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate the metrics that go into the daily Discord summary.

    Pure-ish — depends on DB state but doesn't notify anyone. Returns a
    flat dict ready to splat into `notify(...)` as embed fields.
    """
    now = now or datetime.now(UTC).replace(tzinfo=None)
    today_start = datetime.combine(now.date(), datetime.min.time())
    week_start = (now - timedelta(days=7))
    account_id = config.get("account", {}).get("id", "primary")

    c = await repos.db.connect()

    # Today's closed cycles
    async with c.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(final_pnl), 0) AS pnl "
        "FROM wheel_cycles WHERE account_id = ? AND ended_at >= ? "
        "AND final_pnl IS NOT NULL",
        (account_id, today_start.isoformat()),
    ) as cur:
        row = await cur.fetchone()
    today_closed = int(row["n"]) if row else 0
    today_pnl = float(row["pnl"]) if row and row["pnl"] is not None else 0.0

    # Today's cycle starts
    async with c.execute(
        "SELECT COUNT(*) AS n FROM wheel_cycles "
        "WHERE account_id = ? AND started_at >= ?",
        (account_id, today_start.isoformat()),
    ) as cur:
        row = await cur.fetchone()
    today_opened = int(row["n"]) if row else 0

    # Week-to-date (rolling 7 days) closed cycles
    async with c.execute(
        "SELECT COUNT(*) AS n, "
        "       COALESCE(SUM(final_pnl), 0) AS pnl, "
        "       SUM(CASE WHEN final_pnl > 0 THEN 1 ELSE 0 END) AS wins "
        "FROM wheel_cycles WHERE account_id = ? AND ended_at >= ? "
        "AND final_pnl IS NOT NULL",
        (account_id, week_start.isoformat()),
    ) as cur:
        row = await cur.fetchone()
    week_closed = int(row["n"]) if row else 0
    week_pnl = float(row["pnl"]) if row and row["pnl"] is not None else 0.0
    week_wins = int(row["wins"] or 0) if row else 0
    win_rate_pct = (week_wins / week_closed * 100.0) if week_closed > 0 else 0.0

    # Open positions per strategy (excludes IDLE)
    async with c.execute(
        "SELECT strategy_id, COUNT(*) AS n FROM positions "
        "WHERE account_id = ? AND state != 'IDLE' "
        "GROUP BY strategy_id ORDER BY strategy_id",
        (account_id,),
    ) as cur:
        rows = await cur.fetchall()
    open_by_strategy = {r["strategy_id"]: int(r["n"]) for r in rows}
    open_total = sum(open_by_strategy.values())

    # Latest regime
    async with c.execute(
        "SELECT snapshot_date, regime, csps_allowed, bear_calls_allowed "
        "FROM regime_snapshots ORDER BY snapshot_date DESC LIMIT 1"
    ) as cur:
        regime_row = await cur.fetchone()
    regime_label = regime_row["regime"] if regime_row else "—"
    csps_flag = (
        "✓" if regime_row and regime_row["csps_allowed"] else
        ("✗" if regime_row else "—")
    )
    calls_flag = (
        "✓" if regime_row and regime_row["bear_calls_allowed"] else
        ("✗" if regime_row else "—")
    )

    # Auto-disabled strategies (auto-clears stale rows as side effect)
    disabled = await repos.strategy_runtime.list_disabled(now=now)
    disabled_summary = (
        ", ".join(f"{r['strategy_id']}" for r in disabled) if disabled else "none"
    )

    return {
        "date": now.date().isoformat(),
        "today_pnl": f"${today_pnl:+.2f}",
        "today_closed": today_closed,
        "today_opened": today_opened,
        "week_pnl": f"${week_pnl:+.2f}",
        "week_closed": week_closed,
        "week_win_rate": f"{win_rate_pct:.0f}%" if week_closed > 0 else "—",
        "open_total": open_total,
        "open_by_strategy": ", ".join(
            f"{s}={n}" for s, n in sorted(open_by_strategy.items())
        ) or "none",
        "regime": regime_label,
        "csps_allowed": csps_flag,
        "bear_calls_allowed": calls_flag,
        "auto_disabled": disabled_summary,
    }


async def main_async() -> int:
    config = load_config()
    set_dispatcher(make_notifier(config))
    db_path = Path(config.get("database", {}).get("path", "wheelbot.db")).expanduser()
    async with Database(db_path) as db:
        repos = Repos(db)
        metrics = await compute_metrics(repos, config)
    title = f"WheelBot Daily Summary — {metrics['date']}"
    await notify("daily_summary", title, **metrics)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
