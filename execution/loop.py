"""Reconciler loop runner.

Composes Reconciler + KillSwitch + cadence. The loop reconciles every tick;
the kill switch gates whether to *propose new orders* on that tick (the
reconciler runs regardless — that's the spec's contract).

Cadence:
    in market hours → `reconciler.market_hours_interval_seconds` (default 300)
    off-hours       → `reconciler.off_hours_interval_seconds`    (default 1800)

Failure handling:
    Consecutive Reconciler exceptions ≥ `max_failures_before_pause` (default 5)
    flips active positions to BROKER_DOWN. This module does not surface
    BROKER_DOWN positions back to *_OPEN — once down, an operator inspects.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import PositionState
from core.notify import notify
from db.repo import Repos
from execution.kill_switch import KillSwitch, KillSwitchResult
from execution.market_hours import is_market_hours
from execution.reconciler import Reconciler


class ReconcilerLoop:
    def __init__(
        self,
        broker: Broker,
        repos: Repos,
        config: dict[str, Any],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        reconciler: Reconciler | None = None,
        post_tick: Callable[["ReconcilerLoop", "KillSwitchResult | None"], Awaitable[None]] | None = None,
    ) -> None:
        self._broker = broker
        self._repos = repos
        self._config = config
        self._sleep = sleep
        # Production callers pass an injected Reconciler so they can wire a
        # roll_evaluator etc.; tests/the simple case construct one inline.
        self._reconciler = reconciler or Reconciler(broker, repos, config)
        self._kill_switch = KillSwitch(broker, repos, config)
        self._post_tick = post_tick
        self._consecutive_failures = 0
        section = config.get("reconciler", {})
        self._market_interval = float(section.get("market_hours_interval_seconds", 300))
        self._off_interval = float(section.get("off_hours_interval_seconds", 1800))
        self._max_failures = int(section.get("max_failures_before_pause", 5))
        self._account_id = config.get("account", {}).get("id", "primary")

    async def tick(self) -> tuple[Reconciler, KillSwitchResult | None]:
        """One reconcile + kill-switch evaluation. Returns the kill-switch
        result so the caller (e.g. router orchestrator) can gate on it."""
        try:
            await self._kill_switch.prime_session()
            summary = await self._reconciler.reconcile_once()
            self._consecutive_failures = 0
            log_checkpoint(
                "loop_tick",
                status="ok",
                fills=summary.fills_processed,
                manual=summary.manual_interventions,
            )
        except Exception as exc:
            self._consecutive_failures += 1
            log_checkpoint(
                "loop_tick",
                status="fail",
                error=str(exc),
                consecutive_failures=self._consecutive_failures,
            )
            if self._consecutive_failures >= self._max_failures:
                await self._mark_broker_down()
            return self._reconciler, None

        ks = await self._kill_switch.check()
        if self._post_tick is not None:
            try:
                await self._post_tick(self, ks)
            except Exception as exc:
                log_checkpoint(
                    "loop_post_tick_fail",
                    status="fail",
                    error=str(exc),
                )
        return self._reconciler, ks

    async def run_forever(self) -> None:
        while True:
            await self.tick()
            interval = self._market_interval if is_market_hours() else self._off_interval
            await self._sleep(interval)

    async def _mark_broker_down(self) -> None:
        """Set every active local position to BROKER_DOWN. Operator inspects."""
        active = await self._repos.positions.list_active(self._account_id)
        from datetime import UTC, datetime

        now = datetime.now(UTC).replace(tzinfo=None)
        for pos in active:
            if pos.id is None:
                continue
            await self._repos.positions.update_state(
                pos.id,
                PositionState.BROKER_DOWN,
                f"loop:{self._consecutive_failures} consecutive failures",
                when=now,
            )
        log_checkpoint(
            "loop_broker_down",
            status="fail",
            n_positions=len(active),
            consecutive_failures=self._consecutive_failures,
        )
        await notify(
            "position.broker_down",
            "Broker connection failing — positions paused",
            n_positions=len(active),
            consecutive_failures=self._consecutive_failures,
        )
