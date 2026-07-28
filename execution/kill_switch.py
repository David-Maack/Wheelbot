"""Kill switch — rules 8-10 of spec §8.

These are runaway-stops, not bad-trade-stops. Run this BEFORE proposing/placing
new orders on each loop tick. The reconciler keeps running regardless.

Rules:

  8. Daily P&L drawdown — `daily_loss_kill_switch_pct` (default 5%) of equity
     from session-open. Anchor stored in `daily_state` so a mid-day restart
     keeps the baseline.
  9. Consecutive losing cycles — N closed cycles with `final_pnl < 0` in a row.
 10. Manual stop file — existence of `risk.stop_file_path` halts new orders.

A tripped switch sets `daily_state.kill_switch_armed = True` so a mid-day
operator can see why we stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import DailyState
from core.notify import notify
from db.repo import Repos


@dataclass(frozen=True, slots=True)
class KillSwitchResult:
    tripped: bool
    reasons: list[str]

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "ok"


class KillSwitch:
    def __init__(
        self,
        broker: Broker,
        repos: Repos,
        config: dict[str, Any],
    ) -> None:
        self._broker = broker
        self._repos = repos
        self._config = config

    async def prime_session(
        self,
        *,
        today: date | None = None,
    ) -> DailyState:
        """Capture session-open equity if not already set for today.

        Idempotent — calling on an already-primed day is a no-op.
        Call this from the first loop tick of the day.
        """
        today = today or datetime.now(UTC).date()
        account_id = self._config.get("account", {}).get("id", "primary")
        existing = await self._repos.daily_state.get(account_id, today)
        if existing and existing.session_open_equity is not None:
            return existing
        account = await self._broker.get_account()
        entry = DailyState(
            account_id=account_id,
            snapshot_date=today,
            session_open_equity=account.equity,
            consecutive_losses=existing.consecutive_losses if existing else 0,
            kill_switch_armed=existing.kill_switch_armed if existing else False,
            kill_switch_reason=existing.kill_switch_reason if existing else None,
        )
        await self._repos.daily_state.upsert(entry)
        log_checkpoint(
            "kill_switch_prime",
            status="ok",
            account_id=account_id,
            equity=account.equity,
        )
        return entry

    async def check(self, *, today: date | None = None) -> KillSwitchResult:
        today = today or datetime.now(UTC).date()
        account_id = self._config.get("account", {}).get("id", "primary")
        reasons: list[str] = []

        # Rule 10 — manual stop file.
        stop_file = self._config.get("risk", {}).get("stop_file_path")
        if stop_file and Path(stop_file).expanduser().exists():
            reasons.append(f"manual stop file present: {stop_file}")

        # Rule 8 — daily P&L drawdown.
        loss_pct_max = float(self._config.get("risk", {}).get("daily_loss_kill_switch_pct", 5))
        anchor = await self._repos.daily_state.get(account_id, today)
        if anchor is None or anchor.session_open_equity is None:
            log_checkpoint(
                "kill_switch_no_anchor",
                status="skip",
                detail="daily P&L gate cannot evaluate without session-open equity",
            )
        else:
            account = await self._broker.get_account()
            drawdown_pct = (
                (anchor.session_open_equity - account.equity)
                / anchor.session_open_equity
                * 100.0
            )
            if drawdown_pct > loss_pct_max:
                reasons.append(
                    f"daily drawdown {drawdown_pct:.2f}% > {loss_pct_max}%"
                )

        # Rule 9 — consecutive losing cycles.
        max_losses = int(self._config.get("risk", {}).get("consecutive_losses_pause", 3))
        losses = await self._consecutive_losses(account_id, max_losses)
        if losses >= max_losses:
            reasons.append(f"{losses} consecutive losing cycles >= {max_losses}")

        # 2026-07-23 review fix: SESSION LATCH. Before this, tripped was
        # recomputed fresh every tick — equity bouncing from -5.1% to -4.9%
        # DISARMED the daily-loss stop mid-session and trading resumed. An
        # automatic trip (drawdown / consecutive losses) now stays tripped
        # for the rest of the snapshot_date. Manual stop-file trips are NOT
        # latched — removing the file (scripts or MCP release_kill_switch)
        # must release, and the MCP release also clears this latch so the
        # operator always has the override.
        if not reasons and anchor and anchor.kill_switch_armed:
            prior = anchor.kill_switch_reason or ""
            if "drawdown" in prior or "consecutive" in prior:
                reasons.append(f"latched for the session: {prior}")

        # Persist arming state.
        if reasons or (anchor and anchor.kill_switch_armed):
            await self._repos.daily_state.upsert(
                DailyState(
                    account_id=account_id,
                    snapshot_date=today,
                    session_open_equity=anchor.session_open_equity if anchor else None,
                    consecutive_losses=losses,
                    kill_switch_armed=bool(reasons),
                    kill_switch_reason="; ".join(reasons) if reasons else None,
                )
            )

        result = KillSwitchResult(tripped=bool(reasons), reasons=reasons)
        log_checkpoint(
            "kill_switch_check",
            status="fail" if result.tripped else "ok",
            tripped=result.tripped,
            reasons=reasons,
        )
        # Edge-trigger: only notify on the transition into armed (avoid spamming
        # every tick once tripped). `anchor.kill_switch_armed` reflects the
        # state going INTO this check; we now know whether to write True.
        was_armed = bool(anchor.kill_switch_armed) if anchor else False
        if result.tripped and not was_armed:
            await notify(
                "risk.kill_switch_armed",
                "Kill switch armed",
                account_id=account_id,
                reasons="; ".join(reasons),
            )
        return result

    async def _consecutive_losses(self, account_id: str, look: int) -> int:
        """Count closed cycles in a row (most-recent-first) with final_pnl < 0."""
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT final_pnl FROM wheel_cycles "
            "WHERE account_id = ? AND ended_at IS NOT NULL AND final_pnl IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT ?",
            (account_id, max(look + 1, 5)),
        ) as cur:
            rows = await cur.fetchall()
        streak = 0
        for row in rows:
            if row["final_pnl"] < 0:
                streak += 1
            else:
                break
        return streak
