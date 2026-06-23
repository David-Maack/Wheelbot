"""Reconciliation loop — single source of truth for state-after-fill.

Per spec §10:

  Bot can crash, broker can lag, fills can happen overnight (assignments). The
  reconciler is the source of truth.

  Runs every 5 minutes during market hours, every 30 off-hours.

  For each account_id:
    1. Fetch current positions from broker.
    2. Fetch all orders updated since last reconcile.
    3. Compare to local DB:
       - New fill → update order, transition state, log.
       - Assignment → CSP_OPEN → SHARES_HELD, cost_basis = strike - premium.
       - Worthless expiration → CSP_OPEN → IDLE (or CC_OPEN → SHARES_HELD), close
         cycle if applicable.
       - Called away → CC_OPEN → IDLE, close cycle, calc realized P&L.
       - Mismatch (broker shows position we don't have, or vice versa) → flag
         MANUAL_INTERVENTION, do NOT auto-correct.
    4. Recompute cost_basis after every state change.

  The reconciler is the only thing that writes state changes for fills. The order
  router only writes PENDING. This separation prevents double-counting and means
  bot crashes mid-order are recoverable.

This module is the most heavily-tested. Every transition path needs a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from core.broker import Broker
from core.checkpoint import checkpoint, log_checkpoint
from core.models import (
    CycleOutcome,
    Order,
    OrderStatus,
    OrderType,
    OptionType,
    Position,
    PositionState,
    StateLog,
    StateLogTrigger,
    WheelCycle,
)
from core.notify import notify
from db.repo import Repos


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _observed_partial_fill(local: Order, broker_view: Order) -> bool:
    """True if some quantity executed before the order terminated.

    Two signals, in priority order:
      1. We persisted PARTIAL on a prior tick (the reliable, broker-agnostic
         signal — the reconcile loop writes status every tick).
      2. Best-effort: the broker's raw payload carries a positive `filled_qty`
         below the ordered quantity (catches a partial-then-cancel that
         collapsed into a single tick, e.g. EOD auto-cancel of a DAY order).
    """
    if local.status == OrderStatus.PARTIAL:
        return True
    raw = broker_view.raw_response or {}
    try:
        filled = float(raw.get("filled_qty") or 0)
    except (TypeError, ValueError):
        return False
    return 0 < filled < float(broker_view.quantity or local.quantity or 0)


# When a cycle is closed with a "profit" outcome by the caller but the realized
# P&L is actually negative (e.g. a stop-loss / defensive buyback hardcoded to
# CSP_CLOSED_PROFIT), relabel to the matching loss outcome so the dashboard's
# Outcomes breakdown is honest. The win/loss tally keys off final_pnl sign, so
# it's already correct once P&L is right — this just fixes the label.
_PROFIT_TO_LOSS_OUTCOME = {
    CycleOutcome.CSP_CLOSED_PROFIT: CycleOutcome.CSP_CLOSED_LOSS,
    CycleOutcome.CC_CLOSED_PROFIT: CycleOutcome.CC_CLOSED_LOSS,
    CycleOutcome.SPREAD_CLOSED_PROFIT: CycleOutcome.SPREAD_CLOSED_LOSS,
    # TICKET-014: iron condor uses its own outcome family so post-hoc
    # filtering on /cycles can split iron_condor results from vanilla
    # vertical spreads.
    CycleOutcome.IRON_CONDOR_CLOSED_PROFIT: CycleOutcome.IRON_CONDOR_CLOSED_LOSS,
    # TICKET-016: calendar.
    CycleOutcome.CALENDAR_CLOSED_PROFIT: CycleOutcome.CALENDAR_CLOSED_LOSS,
}


def _spread_close_outcome(strategy_id: str | None) -> CycleOutcome:
    """Pick the right CycleOutcome at MULTI_LEG_CLOSE. iron_condor and calendar
    cycles get tagged distinctly from vanilla spreads so the /cycles filter and
    /performance breakdown can separate them."""
    if strategy_id == "iron_condor":
        return CycleOutcome.IRON_CONDOR_CLOSED_PROFIT
    if strategy_id == "calendar":
        return CycleOutcome.CALENDAR_CLOSED_PROFIT
    return CycleOutcome.SPREAD_CLOSED_PROFIT


def _spread_expired_outcome(strategy_id: str | None) -> CycleOutcome:
    """Same dispatch for MULTI_LEG_OPEN expiration paths (all legs OTM)."""
    if strategy_id == "iron_condor":
        return CycleOutcome.IRON_CONDOR_EXPIRED_PROFIT
    if strategy_id == "calendar":
        return CycleOutcome.CALENDAR_EXPIRED
    return CycleOutcome.SPREAD_EXPIRED_PROFIT


# TICKET-014.5: PositionStates that _diff_one INTENTIONALLY does not act on —
# either terminal (CSP_CLOSED, CALLED_AWAY, ...), operator-owned
# (MANUAL_INTERVENTION, KILLED), transient between fills (IDLE, SCANNING,
# ROLL_EVAL, ASSIGNED), or driven by _on_fill rather than position-shape
# (the *_PENDING states are handled explicitly above; SHARES_HELD has no
# shape-only transition today). The complement of this set and the states
# _diff_one explicitly branches on must cover EVERY PositionState — enforced
# by test_diff_one_covers_every_position_state. When a new state is added
# (e.g. PMCC_* in TICKET-015) it must be categorized here or given a branch,
# otherwise the test fails AND _diff_one flags it MANUAL_INTERVENTION at
# runtime rather than silently skipping it.
_DIFF_ONE_NO_TRANSITION_STATES: frozenset[PositionState] = frozenset({
    PositionState.IDLE,
    PositionState.SCANNING,
    PositionState.ROLL_EVAL,
    PositionState.CSP_CLOSED,
    PositionState.ASSIGNED,
    PositionState.SHARES_HELD,
    PositionState.CC_CLOSED,
    PositionState.CALLED_AWAY,
    PositionState.SPREAD_CLOSED,
    PositionState.SPREAD_ASSIGNED,
    # TICKET-015 PMCC: the two *_PENDING states are driven by _on_fill
    # (PENDING → OPEN on observed fill); CLOSING is the transient full-unwind
    # state driven by _on_fill of the long SELL_TO_CLOSE. The two ACTIVE PMCC
    # states (PMCC_LONG_OPEN, PMCC_BOTH_OPEN) get explicit _diff_one branches.
    PositionState.PMCC_LONG_PENDING,
    PositionState.PMCC_SHORT_PENDING,
    PositionState.PMCC_CLOSING,
    PositionState.BROKER_DOWN,
    PositionState.MANUAL_INTERVENTION,
    PositionState.KILLED,
})

# TICKET-014.5: order types whose P&L multiplier _compute_cycle_pnl knows.
# Options trade at 100×; the synthetic stock legs written by assignment /
# called-away (BUY_TO_OPEN / SELL_TO_CLOSE with option_type=None) trade at 1×.
# A BUY_TO_OPEN / SELL_TO_CLOSE that DOES carry an option_type is an option
# leg (PMCC's long call is the first such case) — priced at 100× and logged.
_PNL_OPTION_ORDER_TYPES: frozenset[OrderType] = frozenset({
    OrderType.SELL_TO_OPEN,
    OrderType.BUY_TO_CLOSE,
})
_PNL_STOCK_OR_OPTION_ORDER_TYPES: frozenset[OrderType] = frozenset({
    OrderType.BUY_TO_OPEN,
    OrderType.SELL_TO_CLOSE,
})


@dataclass(slots=True)
class ReconcileSummary:
    fills_processed: int = 0
    cancellations_processed: int = 0
    expirations_processed: int = 0
    assignments_processed: int = 0
    called_aways_processed: int = 0
    manual_interventions: int = 0
    cycles_closed: int = 0
    cycles_opened: int = 0
    rolls_evaluated: int = 0
    transitions: list[tuple[str, PositionState, PositionState, str]] = field(default_factory=list)


class Reconciler:
    def __init__(
        self,
        broker: Broker,
        repos: Repos,
        config: dict[str, Any],
        *,
        roll_evaluator: Any = None,
        universe: dict[str, Any] | None = None,
    ) -> None:
        self._broker = broker
        self._repos = repos
        self._config = config
        self._account_id = config.get("account", {}).get("id", "primary")
        # Optional callback `(position, short_order, current_quote) -> RollOutcome`.
        # When provided, the reconciler runs the roll trigger scan after the
        # standard fill/expiration/assignment sweep. Tests pass a stub.
        self._roll_evaluator = roll_evaluator
        self._universe = universe or {"tickers": [], "banned": [], "banned_rules": []}
        # In-memory cursor for "orders updated since"; persists across reconcile_once
        # calls within a single Reconciler instance. Loop runner re-uses the
        # instance so this stays tight.
        self._orders_cursor: datetime | None = None

    async def reconcile_once(self) -> ReconcileSummary:
        summary = ReconcileSummary()
        with checkpoint("reconcile_once", account_id=self._account_id) as ctx:
            broker_positions = await self._broker.get_positions()
            # Cursor is the upper bound on what we've already processed, but
            # we MUST always include any in-flight (PENDING/PARTIAL) orders so
            # we catch their eventual FILLED status. Alpaca's get_orders_since
            # filters by submitted_at, not updated_at — so an order submitted
            # before the cursor would otherwise be invisible to subsequent ticks.
            since = self._orders_cursor or datetime(2000, 1, 1)
            oldest_pending = await self._repos.orders.oldest_pending_placed_at(
                self._account_id
            )
            if oldest_pending is not None and oldest_pending < since:
                # Hold the lookback window to include the oldest pending order
                # (small safety margin for clock skew).
                since = oldest_pending - timedelta(seconds=5)
            broker_orders = await self._broker.get_orders_since(since)
            self._orders_cursor = _utcnow()

            await self._process_orders(broker_orders, summary)
            await self._reconcile_positions(broker_positions, summary)
            if self._roll_evaluator is not None:
                await self._scan_roll_triggers(summary)

            ctx["fills"] = summary.fills_processed
            ctx["assignments"] = summary.assignments_processed
            ctx["called_aways"] = summary.called_aways_processed
            ctx["expirations"] = summary.expirations_processed
            ctx["manual_interventions"] = summary.manual_interventions
            ctx["rolls_evaluated"] = summary.rolls_evaluated
        return summary

    async def _scan_roll_triggers(self, summary: ReconcileSummary) -> None:
        """For each open short option, check if the trigger is met. The actual
        evaluation is the orchestrator's job — passed in via roll_evaluator."""
        active = await self._repos.positions.list_active(self._account_id)
        for pos in active:
            if pos.id is None:
                continue
            state = pos.state.value if hasattr(pos.state, "value") else str(pos.state)
            if state not in ("CSP_OPEN", "CC_OPEN"):
                continue
            short = await self._latest_short_order_for_position(pos)
            if short is None or short.contract_symbol is None:
                continue
            try:
                quote = await self._broker.get_quote(short.contract_symbol)
            except Exception:
                continue
            mid = quote.mid if quote.mid is not None else (quote.last or 0.0)
            try:
                outcome = await self._roll_evaluator(pos, short, mid)
            except Exception as exc:
                log_checkpoint(
                    "roll_eval_fail", status="fail", symbol=pos.symbol, error=str(exc)
                )
                continue
            if outcome is None or getattr(outcome, "action", None) is None:
                continue
            summary.rolls_evaluated += 1

    async def _latest_short_order_for_position(self, pos: Position) -> Order | None:
        if pos.current_cycle_id is None:
            return None
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ? "
            "AND status = ? ORDER BY placed_at DESC LIMIT 1",
            (
                pos.current_cycle_id,
                OrderType.SELL_TO_OPEN.value,
                OrderStatus.FILLED.value,
            ),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE

        return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))

    # -- Order processing -----------------------------------------------------

    async def _process_orders(
        self,
        broker_orders: list[Order],
        summary: ReconcileSummary,
    ) -> None:
        for broker_view in broker_orders:
            if broker_view.client_order_id is None and broker_view.broker_order_id is None:
                continue
            local = None
            if broker_view.client_order_id:
                local = await self._repos.orders.get_by_client_id(broker_view.client_order_id)
            if local is None and broker_view.broker_order_id:
                local = await self._repos.orders.get_by_broker_id(broker_view.broker_order_id)
            if local is None:
                # Order at broker we don't know about — flag the affected position.
                await self._flag_manual_intervention(
                    broker_view.symbol,
                    f"unknown order {broker_view.broker_order_id} at broker",
                    summary,
                )
                continue
            if local.id is None:
                continue

            # Persist any new fields broker has but DB doesn't.
            updates: dict[str, Any] = {}
            if broker_view.broker_order_id and broker_view.broker_order_id != local.broker_order_id:
                updates["broker_order_id"] = broker_view.broker_order_id
            if broker_view.status != local.status:
                updates["status"] = broker_view.status.value if hasattr(broker_view.status, "value") else str(broker_view.status)
            if broker_view.fill_price is not None and broker_view.fill_price != local.fill_price:
                updates["fill_price"] = broker_view.fill_price
            if broker_view.filled_at is not None and broker_view.filled_at != local.filled_at:
                updates["filled_at"] = broker_view.filled_at.isoformat()
            if broker_view.raw_response is not None and broker_view.raw_response != local.raw_response:
                updates["raw_response"] = broker_view.raw_response
            if updates:
                await self._repos.orders.update(local.id, **updates)

            # Handle terminal status transitions.
            local_was_pending = local.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)
            now_filled = broker_view.status == OrderStatus.FILLED
            now_cancelled = broker_view.status in (
                OrderStatus.CANCELLED, OrderStatus.REJECTED,
            )
            # TICKET-014.5: per-order isolation. An exception in _on_fill /
            # _on_cancel (a forgotten branch that raises, a KeyError deep in a
            # handler) previously propagated up through reconcile_once and
            # aborted the ENTIRE tick — halting reconciliation for every other
            # healthy position. Wrap each order's dispatch so one bad order
            # flags MANUAL_INTERVENTION and the tick continues. Combined with
            # the explicit unhandled-type guard in _on_fill, this makes the
            # reconcile loop fail loud-but-isolated rather than silent-or-total.
            try:
                await self._dispatch_order_transition(
                    local, broker_view, local_was_pending, now_filled,
                    now_cancelled, summary,
                )
            except Exception as exc:  # noqa: BLE001 — deliberate catch-all
                log_checkpoint(
                    "reconcile_order_dispatch_error",
                    status="fail",
                    symbol=broker_view.symbol,
                    client_order_id=local.client_order_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                await self._flag_manual_intervention(
                    broker_view.symbol,
                    f"reconcile error on order {local.client_order_id}: "
                    f"{type(exc).__name__}: {exc}",
                    summary,
                )

    async def _dispatch_order_transition(
        self,
        local: Order,
        broker_view: Order,
        local_was_pending: bool,
        now_filled: bool,
        now_cancelled: bool,
        summary: ReconcileSummary,
    ) -> None:
        """Route one order's terminal-status transition. Extracted from
        _process_orders so the per-order try/except (TICKET-014.5) wraps a
        single call site."""
        if local_was_pending and now_filled:
            await self._on_fill(local, broker_view, summary)
        elif local_was_pending and now_cancelled:
            # A DAY order that partially filled and then cancelled (or got
            # cancelled at EOD with some contracts/legs already executed)
            # leaves REAL filled quantity live at the broker. _on_cancel
            # would reset the position as if nothing filled, orphaning those
            # contracts. Hand it to a human instead of corrupting state.
            if _observed_partial_fill(local, broker_view):
                await self._flag_manual_intervention(
                    local.symbol,
                    f"order {local.broker_order_id} {broker_view.status.value} "
                    f"after partial fill — manual reconcile",
                    summary,
                )
            else:
                await self._on_cancel(local, broker_view, summary)

    async def _on_fill(
        self,
        local: Order,
        broker_view: Order,
        summary: ReconcileSummary,
    ) -> None:
        summary.fills_processed += 1
        symbol = local.symbol
        # Multi-leg orders are scoped per-strategy because the same underlying
        # can run a wheel CSP and a put_spread on independent positions.
        position = await self._repos.positions.get_by_symbol(
            self._account_id, symbol, strategy_id=local.strategy_id
        )
        fill_price = broker_view.fill_price if broker_view.fill_price is not None else (local.limit_price or 0.0)

        # TICKET-015: PMCC fills follow their own lifecycle — the long LEAP
        # persists across many short calls within ONE cycle. Dispatch them
        # before the wheel/spread order-type chain so the wheel CSP/CC path
        # below stays byte-identical (the strategy_id guard is the firewall).
        if local.strategy_id == "pmcc":
            await self._on_pmcc_fill(local, position, fill_price, summary)
            return

        if local.order_type == OrderType.MULTI_LEG_OPEN:
            cycle_id = await self._open_cycle_for_spread(local, fill_price, summary)
            await self._set_position_state(
                position,
                symbol,
                PositionState.SPREAD_OPEN,
                f"fill:{local.client_order_id}",
                cycle_id=cycle_id or (position.current_cycle_id if position else None),
                strategy_id=local.strategy_id,
            )
            if cycle_id is not None and local.id is not None:
                await self._repos.orders.update(local.id, cycle_id=cycle_id)
            return

        if local.order_type == OrderType.MULTI_LEG_CLOSE:
            await self._set_position_state(
                position,
                symbol,
                PositionState.SPREAD_CLOSED,
                f"close:{local.client_order_id}",
                strategy_id=local.strategy_id,
            )
            if position and position.current_cycle_id is not None:
                # Tag the close order with the cycle so PnL captures both legs.
                if local.id is not None and local.cycle_id is None:
                    await self._repos.orders.update(local.id, cycle_id=position.current_cycle_id)
                await self._close_cycle(
                    position.current_cycle_id,
                    _spread_close_outcome(local.strategy_id),
                    summary,
                )
                # Clear the cycle pointer — the spread is closed and the
                # position is now between cycles. Without this, the dashboard
                # computes phantom P&L off the dead cycle's legs while a new
                # spread open is in flight (position SPREAD_PENDING). Set to
                # None so derived columns show "—" until a new open fills.
                if position.id is not None:
                    await self._repos.positions.update(position.id, current_cycle_id=None)
            return

        if local.order_type == OrderType.SELL_TO_OPEN:
            # CSP_PENDING → CSP_OPEN, or CC_PENDING → CC_OPEN.
            is_put = local.option_type == OptionType.PUT
            new_state = PositionState.CSP_OPEN if is_put else PositionState.CC_OPEN
            # Puts open a new wheel cycle; calls (CCs) continue the EXISTING
            # cycle the position is already in (assignment → CC is one cycle).
            existing_cycle = position.current_cycle_id if position else None
            if is_put:
                cycle_id = await self._open_cycle_if_csp(local, fill_price, summary)
                effective_cycle = cycle_id or existing_cycle
            else:
                cycle_id = None
                effective_cycle = existing_cycle
            await self._set_position_state(
                position,
                symbol,
                new_state,
                f"fill:{local.client_order_id}",
                cycle_id=effective_cycle,
            )
            # Tag the order with its cycle so the dashboard + P&L accounting
            # can find it. Previously only puts that *opened* a cycle got
            # tagged, so CC orders had cycle_id=NULL and the dashboard fell
            # back to the (expired) CSP for DTE / unrealized.
            if effective_cycle is not None and local.id is not None:
                await self._repos.orders.update(local.id, cycle_id=effective_cycle)

        elif local.order_type == OrderType.BUY_TO_CLOSE:
            # The reconciler infers the resulting state from the contract type +
            # whether shares are still held. CSP close → IDLE (cycle continues
            # only if assignment didn't already happen — but if we're buying back
            # the put, no assignment); CC close → SHARES_HELD.
            is_put = local.option_type == OptionType.PUT
            new_state = PositionState.IDLE if is_put else PositionState.SHARES_HELD
            # Tag the buyback with the cycle BEFORE _close_cycle computes P&L.
            # Without this, _compute_cycle_pnl (which filters on cycle_id) never
            # sees the buyback debit and books the FULL premium as profit —
            # overstating every profit-close / time-close / stop-loss exit and
            # mislabeling losses as wins.
            if (
                position is not None
                and position.current_cycle_id is not None
                and local.id is not None
                and local.cycle_id is None
            ):
                await self._repos.orders.update(
                    local.id, cycle_id=position.current_cycle_id
                )
            await self._set_position_state(
                position,
                symbol,
                new_state,
                f"close:{local.client_order_id}",
            )
            if is_put and position and position.current_cycle_id is not None:
                await self._close_cycle(
                    position.current_cycle_id,
                    CycleOutcome.CSP_CLOSED_PROFIT,
                    summary,
                )

        else:
            # TICKET-014.5: explicit unhandled-order-type guard. Before this,
            # a filled BUY_TO_OPEN / SELL_TO_CLOSE (or any future order type)
            # fell through silently — no state set, no cycle opened, invisible.
            # PMCC's long-call BUY_TO_OPEN is the next such case; surfacing it
            # loudly (flag + checkpoint) means the omission in TICKET-015 can't
            # corrupt state quietly. Flag rather than raise so one unhandled
            # order doesn't abort reconciliation for healthy positions (the
            # per-item isolation in _process_orders is the second backstop).
            log_checkpoint(
                "reconcile_unhandled_order_type",
                status="fail",
                symbol=symbol,
                strategy=local.strategy_id,
                order_type=local.order_type.value if hasattr(local.order_type, "value") else str(local.order_type),
                client_order_id=local.client_order_id,
                note="_on_fill has no branch for this order type — add one",
            )
            await self._flag_manual_intervention(
                symbol,
                f"_on_fill: no handler for filled order_type "
                f"{local.order_type} ({local.client_order_id})",
                summary,
            )

    # -- PMCC fill dispatch (TICKET-015) --------------------------------------

    async def _on_pmcc_fill(
        self,
        local: Order,
        position: Position | None,
        fill_price: float,
        summary: ReconcileSummary,
    ) -> None:
        """Route a PMCC fill. The long LEAP is bought first (BUY_TO_OPEN call)
        and persists across many short calls (SELL_TO_OPEN / BUY_TO_CLOSE)
        within ONE cycle. Closing the long (SELL_TO_CLOSE call) ends the cycle.

        Every leg here is an option (option_type set), so _compute_cycle_pnl
        prices them all at 100x via the TICKET-014.5 is_option discriminator —
        PMCC P&L needs no special handling in the P&L computer.
        """
        symbol = local.symbol
        ot = local.order_type
        is_call = local.option_type == OptionType.CALL
        existing_cycle = position.current_cycle_id if position else None

        if ot == OrderType.BUY_TO_OPEN and is_call:
            # Long LEAP opened → open a PMCC cycle → PMCC_LONG_OPEN.
            cycle_id = await self._open_cycle_for_pmcc_long(local, fill_price, summary)
            await self._set_position_state(
                position, symbol, PositionState.PMCC_LONG_OPEN,
                f"fill:{local.client_order_id}",
                cycle_id=cycle_id or existing_cycle,
                strategy_id=local.strategy_id,
            )
            if cycle_id is not None and local.id is not None:
                await self._repos.orders.update(local.id, cycle_id=cycle_id)
            return

        if ot == OrderType.SELL_TO_OPEN and is_call:
            # Short call sold against the long → PMCC_BOTH_OPEN, SAME cycle.
            await self._set_position_state(
                position, symbol, PositionState.PMCC_BOTH_OPEN,
                f"fill:{local.client_order_id}",
                cycle_id=existing_cycle,
                strategy_id=local.strategy_id,
            )
            if existing_cycle is not None and local.id is not None:
                await self._repos.orders.update(local.id, cycle_id=existing_cycle)
            return

        if ot == OrderType.BUY_TO_CLOSE and is_call:
            # Short bought back → PMCC_LONG_OPEN, cycle CONTINUES (the long
            # persists; many shorts per cycle). Tag the buyback so its debit
            # is captured in cycle P&L (mirrors the CC-buyback pattern).
            if existing_cycle is not None and local.id is not None and local.cycle_id is None:
                await self._repos.orders.update(local.id, cycle_id=existing_cycle)
            await self._set_position_state(
                position, symbol, PositionState.PMCC_LONG_OPEN,
                f"close:{local.client_order_id}",
                cycle_id=existing_cycle,
                strategy_id=local.strategy_id,
            )
            return

        if ot == OrderType.SELL_TO_CLOSE and is_call:
            # Long LEAP closed → the cycle ENDS. trigger_reason distinguishes a
            # roll (PMCC_LONG_ROLLED — a new long opens a fresh cycle next) from
            # a full unwind (PMCC_FULL_CLOSED). NOTE: if a roll's new-long open
            # later fails to fill, the cycle is still correctly closed/labelled
            # "rolled" — a cosmetic inaccuracy, not a state bug (you are flat).
            if existing_cycle is not None and local.id is not None and local.cycle_id is None:
                await self._repos.orders.update(local.id, cycle_id=existing_cycle)
            outcome = (
                CycleOutcome.PMCC_LONG_ROLLED
                if (local.trigger_reason or "").startswith("pmcc_roll")
                else CycleOutcome.PMCC_FULL_CLOSED
            )
            await self._set_position_state(
                position, symbol, PositionState.IDLE,
                f"close:{local.client_order_id}",
                strategy_id=local.strategy_id,
            )
            if existing_cycle is not None:
                await self._close_cycle(existing_cycle, outcome, summary)
                if position is not None and position.id is not None:
                    await self._repos.positions.update(position.id, current_cycle_id=None)
            return

        # Unhandled PMCC fill shape (e.g. a PUT, or option_type missing) —
        # flag rather than silently skip.
        log_checkpoint(
            "reconcile_pmcc_unhandled_fill",
            status="fail",
            symbol=symbol,
            order_type=ot.value if hasattr(ot, "value") else str(ot),
            option_type=str(local.option_type),
            client_order_id=local.client_order_id,
        )
        await self._flag_manual_intervention(
            symbol,
            f"_on_pmcc_fill: unhandled fill {ot}/{local.option_type} "
            f"({local.client_order_id})",
            summary,
        )

    async def _open_cycle_for_pmcc_long(
        self, order: Order, fill_price: float, summary: ReconcileSummary,
    ) -> int | None:
        """Open a PMCC cycle on the long-LEAP fill. `fill_price` is the
        per-share debit paid; defined max loss = the debit. Reuses the
        WheelCycle fields (initial_csp_strike = long strike, initial_csp_premium
        = debit paid as a negative cash flow, initial_capital_at_risk = debit)."""
        if order.order_type != OrderType.BUY_TO_OPEN:
            return None
        debit = abs(fill_price)
        qty = order.quantity or 1
        cycle = WheelCycle(
            account_id=self._account_id,
            symbol=order.symbol,
            strategy_id=order.strategy_id,
            started_at=_utcnow(),
            initial_csp_strike=order.strike,
            initial_csp_premium=-debit * 100 * qty,
            initial_capital_at_risk=debit * 100 * qty,
            n_orders=1,
        )
        cycle_id = await self._repos.cycles.insert(cycle)
        summary.cycles_opened += 1
        return cycle_id

    async def _on_pmcc_short_expiration(
        self, local: Position, summary: ReconcileSummary,
    ) -> None:
        """A PMCC short call expired OTM — keep the full premium, return to
        PMCC_LONG_OPEN. The cycle CONTINUES (the long persists); we do NOT
        close it. The short's premium is already captured by its SELL_TO_OPEN
        fill in the cycle's order set, so no synthetic order is needed."""
        summary.expirations_processed += 1
        await self._set_position_state(
            local, local.symbol, PositionState.PMCC_LONG_OPEN,
            "pmcc_short_expired_otm",
            strategy_id=local.strategy_id,
        )

    async def _on_cancel(
        self,
        local: Order,
        broker_view: Order,
        summary: ReconcileSummary,
    ) -> None:
        """Restore position state when an order terminates without filling.

        Handles both single-leg wheel orders and multi-leg spread orders:

          MULTI_LEG_OPEN cancel  → SPREAD_PENDING  → IDLE
          MULTI_LEG_CLOSE cancel → SPREAD_PENDING  → SPREAD_OPEN (still hold legs)
          SELL_TO_OPEN (put) cancel  → CSP_PENDING → IDLE
          SELL_TO_OPEN (call) cancel → CC_PENDING  → SHARES_HELD (still own shares)
          BUY_TO_CLOSE cancel → no-op (router doesn't change position state on
                                close placement for wheels, so no restore needed)

        Defensive: only transitions from the matching *_PENDING state.
        Positions in MANUAL_INTERVENTION, BROKER_DOWN, or any other state
        are left alone — the cancellation is just one of several things that
        could have happened to that position, and the reconciler must not
        clobber a flag set by another rule.
        """
        symbol = local.symbol
        position = await self._repos.positions.get_by_symbol(
            self._account_id, symbol, strategy_id=local.strategy_id
        )
        if position is None:
            return

        target_state: PositionState | None = None
        target_cycle_id: int | None = None
        reason_label = "cancel"

        if local.order_type == OrderType.MULTI_LEG_OPEN:
            if position.state != PositionState.SPREAD_PENDING:
                return
            target_state = PositionState.IDLE
            target_cycle_id = None  # cycle was never opened
            reason_label = "cancel_open"
        elif local.order_type == OrderType.MULTI_LEG_CLOSE:
            if position.state != PositionState.SPREAD_PENDING:
                return
            target_state = PositionState.SPREAD_OPEN
            target_cycle_id = position.current_cycle_id  # cycle still open
            reason_label = "cancel_close"
        elif local.order_type == OrderType.SELL_TO_OPEN:
            # Single-leg wheel entry: CSP_PENDING → IDLE (puts) or
            # CC_PENDING → SHARES_HELD (calls; we still own the underlying).
            is_put = local.option_type == OptionType.PUT
            if is_put and position.state == PositionState.CSP_PENDING:
                target_state = PositionState.IDLE
            elif (not is_put) and position.state == PositionState.CC_PENDING:
                target_state = PositionState.SHARES_HELD
            else:
                return  # state doesn't match what we'd expect for this order
            target_cycle_id = None  # no fill, no cycle opened
            reason_label = "cancel_open"
        else:
            # BUY_TO_CLOSE and other wheel order types: router doesn't move
            # position to a *_PENDING state on placement, so nothing to undo.
            return

        await self._set_position_state(
            position,
            symbol,
            target_state,
            f"{reason_label}:{local.client_order_id}",
            cycle_id=target_cycle_id,
            strategy_id=local.strategy_id,
        )
        summary.cancellations_processed += 1
        log_checkpoint(
            "reconciler_on_cancel",
            status="ok",
            symbol=symbol,
            strategy=local.strategy_id,
            order_type=local.order_type.value if hasattr(local.order_type, "value") else str(local.order_type),
            broker_status=broker_view.status.value if hasattr(broker_view.status, "value") else str(broker_view.status),
            new_state=target_state.value,
        )

    async def _open_cycle_if_csp(
        self,
        order: Order,
        fill_price: float,
        summary: ReconcileSummary,
    ) -> int | None:
        if order.order_type != OrderType.SELL_TO_OPEN or order.option_type != OptionType.PUT:
            return None
        cycle = WheelCycle(
            account_id=self._account_id,
            symbol=order.symbol,
            strategy_id=order.strategy_id,
            started_at=_utcnow(),
            initial_csp_strike=order.strike,
            initial_csp_premium=fill_price * 100 * order.quantity,
            initial_capital_at_risk=(order.strike or 0) * 100 * order.quantity,
            n_orders=1,
        )
        cycle_id = await self._repos.cycles.insert(cycle)
        summary.cycles_opened += 1
        return cycle_id

    async def _open_cycle_for_spread(
        self,
        order: Order,
        fill_price: float,
        summary: ReconcileSummary,
    ) -> int | None:
        """Open a wheel_cycles row for a multi-leg spread.

        The legs live in `order.raw_request['legs']`; we need them to recover
        the package width for capital-at-risk. Falls back to fill_price-based
        accounting if legs are missing (defensive — should always be present).
        """
        if order.order_type != OrderType.MULTI_LEG_OPEN:
            return None
        raw = order.raw_request or {}
        legs = raw.get("legs") or []
        # TICKET-014 precursor #1: prefer the explicit width_dollars stamped
        # by the router when present. For a 4-leg iron condor (90/95P + 105/110C),
        # the legs-only formula (max-strike − min-strike) returns the OUTER
        # span 20 instead of the wing width 5, overstating capital_at_risk ~4×.
        # Verticals stamp the same width so the explicit-vs-derived numbers
        # agree; iron_condor proposals carry the wing width verbatim.
        explicit_width = raw.get("width_dollars")
        if explicit_width is not None:
            width = float(explicit_width)
        else:
            # Fallback for orders placed before the router started stamping
            # width_dollars (defensive — should always be present after the
            # TICKET-014 deploy lands).
            strikes = [leg.get("strike") for leg in legs if leg.get("strike") is not None]
            width = (max(strikes) - min(strikes)) if len(strikes) >= 2 else 0.0
        net_credit_per_share = fill_price  # signed; positive = credit received
        capital_at_risk = max((width - net_credit_per_share) * 100 * order.quantity, 0.0)
        cycle = WheelCycle(
            account_id=self._account_id,
            symbol=order.symbol,
            strategy_id=order.strategy_id,
            started_at=_utcnow(),
            initial_csp_strike=None,
            initial_csp_premium=net_credit_per_share * 100 * order.quantity,
            initial_capital_at_risk=capital_at_risk,
            n_orders=1,
        )
        cycle_id = await self._repos.cycles.insert(cycle)
        summary.cycles_opened += 1
        return cycle_id

    # -- Position reconciliation ---------------------------------------------

    async def _reconcile_positions(
        self,
        broker_positions: list[Position],
        summary: ReconcileSummary,
    ) -> None:
        # A single underlying can have MULTIPLE broker rows at once — most
        # importantly an open covered call (the 100 shares AND the short call
        # are two separate broker positions). Collapsing to one row (last wins)
        # loses the decisive "is the short leg still alive?" signal and made the
        # CC diff fire expiration prematurely / get stuck. Group ALL rows per
        # symbol and let _diff_one reason over the set.
        rows_by_symbol: dict[str, list[Position]] = {}
        for p in broker_positions:
            rows_by_symbol.setdefault(p.symbol.upper(), []).append(p)
        local_positions = await self._repos.positions.list_all(self._account_id)
        local_by_symbol = {p.symbol.upper(): p for p in local_positions}

        # Detect transitions implied purely by *position state* that fills alone
        # don't surface — assignments and worthless expirations.
        # TICKET-014.5: per-symbol isolation — same rationale as the per-order
        # wrap in _process_orders. One symbol's _diff_one raising must not abort
        # reconciliation for every other position.
        for symbol, local in local_by_symbol.items():
            try:
                await self._diff_one(symbol, local, rows_by_symbol.get(symbol, []), summary)
            except Exception as exc:  # noqa: BLE001 — deliberate catch-all
                log_checkpoint(
                    "reconcile_diff_error",
                    status="fail",
                    symbol=symbol,
                    error=f"{type(exc).__name__}: {exc}",
                )
                await self._flag_manual_intervention(
                    symbol,
                    f"reconcile error diffing {symbol}: {type(exc).__name__}: {exc}",
                    summary,
                )

        # Symbols at broker we don't track at all.
        for symbol, rows in rows_by_symbol.items():
            if symbol in local_by_symbol:
                continue
            # Newly discovered position — treat as MANUAL_INTERVENTION rather
            # than guess at state. Spec §10: "do NOT auto-correct."
            rep = rows[0]
            await self._flag_manual_intervention(
                rep.symbol,
                f"broker shows {rep.state} for {rep.symbol} but no local row",
                summary,
            )

    async def _diff_one(
        self,
        symbol: str,
        local: Position,
        broker_rows: list[Position],
        summary: ReconcileSummary,
    ) -> None:
        # Aggregate the broker's view of this underlying. A live short option
        # leg is the decisive signal that a wheel leg is still open — it must
        # override any equity row (an open covered call holds BOTH the shares
        # and the short call simultaneously).
        has_short_put = any(r.state == PositionState.CSP_OPEN for r in broker_rows)
        has_short_call = any(r.state == PositionState.CC_OPEN for r in broker_rows)
        total_shares = sum((r.shares or 0) for r in broker_rows)
        has_shares = total_shares > 0 or any(
            r.state == PositionState.SHARES_HELD for r in broker_rows
        )
        broker_present = bool(broker_rows)

        # Case: local says CSP_OPEN. Short put still alive → still open.
        # Shares appeared → assignment. Nothing left → worthless expiration.
        if local.state == PositionState.CSP_OPEN:
            if has_short_put:
                return  # short put still alive — nothing to do
            if has_shares:
                await self._on_assignment(local, summary)
                return
            await self._on_csp_expiration(local, summary)
            return

        # Case: local says CC_OPEN. Short CALL still alive → still open.
        # Otherwise: shares remain → call expired worthless (keep shares);
        # nothing remains → called away. This mirrors the CSP branch — we key
        # off "is the short leg still alive?" instead of requiring an exact
        # shares+state pair, which previously left the CC stuck at CC_OPEN.
        if local.state == PositionState.CC_OPEN:
            if has_short_call:
                return  # short call still alive — CC still open
            if has_shares:
                await self._on_cc_expiration(local, summary)
                return
            await self._on_called_away(local, summary)
            return

        if local.state in (PositionState.CSP_PENDING, PositionState.CC_PENDING):
            # Normally _on_fill drives PENDING → OPEN off the order stream. But
            # if the reconcile cursor was reset mid-flight (process restart
            # during a rebuild), a one-shot fill/cancel can be missed and the
            # position is stranded in PENDING forever — never managed. Self-heal
            # off broker truth, but only when NO order is still working.
            if await self._has_inflight_order(local):
                return
            if local.state == PositionState.CSP_PENDING:
                if has_short_put:
                    await self._set_position_state(
                        local, symbol, PositionState.CSP_OPEN,
                        "selfheal:stuck_csp_pending",
                        cycle_id=local.current_cycle_id,
                        strategy_id=local.strategy_id,
                    )
                    log_checkpoint(
                        "reconcile_selfheal_pending", status="ok", symbol=symbol,
                        from_state="CSP_PENDING", to_state="CSP_OPEN",
                    )
                elif not broker_present:
                    await self._set_position_state(
                        local, symbol, PositionState.IDLE,
                        "selfheal:csp_pending_unfilled",
                        strategy_id=local.strategy_id,
                    )
                    log_checkpoint(
                        "reconcile_selfheal_pending", status="ok", symbol=symbol,
                        from_state="CSP_PENDING", to_state="IDLE",
                    )
            else:  # CC_PENDING
                if has_short_call:
                    await self._set_position_state(
                        local, symbol, PositionState.CC_OPEN,
                        "selfheal:stuck_cc_pending",
                        cycle_id=local.current_cycle_id,
                        strategy_id=local.strategy_id,
                    )
                    log_checkpoint(
                        "reconcile_selfheal_pending", status="ok", symbol=symbol,
                        from_state="CC_PENDING", to_state="CC_OPEN",
                    )
                elif has_shares:
                    await self._set_position_state(
                        local, symbol, PositionState.SHARES_HELD,
                        "selfheal:cc_pending_unfilled",
                        strategy_id=local.strategy_id,
                    )
                    log_checkpoint(
                        "reconcile_selfheal_pending", status="ok", symbol=symbol,
                        from_state="CC_PENDING", to_state="SHARES_HELD",
                    )
            return

        if local.state == PositionState.SPREAD_OPEN:
            # TICKET-016: a calendar is same-strike, two expirations, NET DEBIT.
            # It must be force-closed before the front leg expires (the 2-DTE
            # close trigger). If we still see it SPREAD_OPEN with the front short
            # leg GONE from the broker, the force-close didn't fire — flag it
            # rather than mishandle the partial-expiry (back leg becomes a naked
            # long). A healthy calendar still shows its short front leg.
            if local.strategy_id == "calendar":
                if not has_short_call and not has_shares:
                    await self._flag_manual_intervention(
                        local.symbol,
                        f"calendar {local.symbol} still SPREAD_OPEN with front "
                        "leg gone — 2-DTE force-close did not fire; review the "
                        "back leg (now an uncovered long)",
                        summary,
                    )
                return
            # Broker shows nothing for this symbol → both legs OTM at expiry,
            # full credit retained. Anything else is asymmetric (one leg ITM,
            # one OTM, or partial assignment) and needs a human eye.
            if not broker_present:
                await self._on_spread_expiration(local, summary)
                return
            # Broker still showing positions on this underlying — could be the
            # spread legs themselves (ok, no action) or shares from assignment
            # of the short put (max-loss path). Stock holdings here are NOT
            # benign: they mean the short put assigned but the long put hasn't
            # been exercised yet — defined-risk-realized territory a human
            # should confirm.
            if has_shares:
                await self._flag_manual_intervention(
                    local.symbol,
                    f"spread {local.symbol} shows shares={total_shares} at broker — "
                    "likely assignment on short leg; review max-loss handling",
                    summary,
                )
            return

        if local.state == PositionState.SPREAD_PENDING:
            # Normally _on_fill (open) / _on_cancel (close) drive PENDING
            # transitions off the order stream. A cursor reset mid-flight
            # (process restart during a rebuild) can drop that one-shot signal,
            # stranding the spread in PENDING forever — the close orchestrator
            # only walks SPREAD_OPEN, so it's never managed and never stopped
            # out (META/GOOGL did exactly this). Self-heal off broker truth, but
            # only when NO order is still working for this position.
            if await self._has_inflight_order(local):
                return
            if broker_present:
                await self._set_position_state(
                    local, symbol, PositionState.SPREAD_OPEN,
                    "selfheal:stuck_spread_pending",
                    cycle_id=local.current_cycle_id,
                    strategy_id=local.strategy_id,
                )
                log_checkpoint(
                    "reconcile_selfheal_pending", status="ok", symbol=symbol,
                    from_state="SPREAD_PENDING", to_state="SPREAD_OPEN",
                )
            else:
                await self._flag_manual_intervention(
                    symbol,
                    f"spread {symbol} stuck SPREAD_PENDING with no in-flight "
                    "order and no legs at broker — open never filled or close "
                    "not reconciled; review",
                    summary,
                )
            return

        # TICKET-015 PMCC. The long LEAP is invisible at the broker on
        # PaperBroker (only short legs are surfaced as rows; real brokers do
        # report longs but never as CC_OPEN/SHARES_HELD). So these branches
        # key on has_short_call / has_shares — robust whether or not the long
        # row is present.
        if local.state == PositionState.PMCC_LONG_OPEN:
            # Expected shape: long alive (invisible) + no short + no shares.
            # Shares appearing here means the long was exercised/assigned or
            # something unexpected happened — hand to a human. A short call
            # row appearing is the short-sale fill, which _on_fill drives.
            if has_shares:
                await self._flag_manual_intervention(
                    local.symbol,
                    f"PMCC {local.symbol} shows shares={total_shares} in "
                    "PMCC_LONG_OPEN — long exercised/assigned? review",
                    summary,
                )
            return

        if local.state == PositionState.PMCC_BOTH_OPEN:
            # Short assignment surfaces as stock at the broker. Rare (the short
            # strike sits above the long's breakeven) and profitable, but the
            # covered-exercise math is NOT auto-reconciled in v1 — flag it (D3).
            if has_shares:
                await self._flag_manual_intervention(
                    local.symbol,
                    f"PMCC {local.symbol} shows shares={total_shares} — short "
                    "call likely assigned; exercise long to cover (manual)",
                    summary,
                )
                return
            if has_short_call:
                return  # short still alive — both legs open
            # No short, no shares → the short call expired worthless. Keep the
            # premium, return to PMCC_LONG_OPEN; the cycle CONTINUES.
            await self._on_pmcc_short_expiration(local, summary)
            return

        # TICKET-014.5: explicit exhaustive guard. States that intentionally
        # imply no position-shape transition are enumerated in
        # _DIFF_ONE_NO_TRANSITION_STATES. Anything that is neither handled
        # above nor in that set is an UNCATEGORIZED state — almost certainly a
        # new PositionState added without wiring it into reconciliation (the
        # PMCC_* states in TICKET-015 are the next such case). Flag it loudly
        # instead of silently skipping, so the gap surfaces as a
        # MANUAL_INTERVENTION + checkpoint rather than a position that never
        # reconciles its expirations/assignments.
        if local.state in _DIFF_ONE_NO_TRANSITION_STATES:
            return
        log_checkpoint(
            "reconcile_uncategorized_state",
            status="fail",
            symbol=symbol,
            strategy=local.strategy_id,
            state=local.state.value if hasattr(local.state, "value") else str(local.state),
            note="_diff_one has no handler and state not in no-transition set — categorize it",
        )
        await self._flag_manual_intervention(
            local.symbol,
            f"_diff_one: uncategorized state {local.state} for {local.symbol} "
            "— reconciliation cannot infer transitions",
            summary,
        )

    async def _on_spread_expiration(
        self, local: Position, summary: ReconcileSummary
    ) -> None:
        """Both legs OTM at expiry → full credit retained, max profit."""
        summary.expirations_processed += 1
        if local.current_cycle_id:
            await self._close_cycle(
                local.current_cycle_id,
                _spread_expired_outcome(local.strategy_id),
                summary,
            )
        if local.id is not None:
            await self._repos.positions.update(
                local.id,
                state=PositionState.IDLE.value,
                shares=0,
                cost_basis=None,
                current_cycle_id=None,
                state_change_reason="spread_expired_max_profit",
                state_changed_at=_utcnow().isoformat(),
            )
            await self._log_state(
                local.id, local.state, PositionState.IDLE, "spread_expired_max_profit"
            )

    async def _on_assignment(self, local: Position, summary: ReconcileSummary) -> None:
        summary.assignments_processed += 1
        await notify(
            "position.assigned",
            f"{local.symbol} assigned",
            symbol=local.symbol,
            cycle_id=local.current_cycle_id,
        )
        # cost_basis = strike - premium_collected_per_share
        cycle = (
            await self._repos.cycles.get(local.current_cycle_id)
            if local.current_cycle_id
            else None
        )
        strike = (cycle.initial_csp_strike if cycle else None) or 0.0
        premium_per_share = (
            (cycle.initial_csp_premium / 100.0)
            if (cycle and cycle.initial_csp_premium)
            else 0.0
        )
        cost_basis = strike - premium_per_share
        # Recover the quantity from the cycle's CSP fill so multi-contract
        # assignments end up with the right share count.
        n_contracts = await self._cycle_csp_quantity(local.current_cycle_id) or 1
        shares = 100 * n_contracts
        if local.id is not None:
            await self._repos.positions.update(
                local.id,
                state=PositionState.SHARES_HELD.value,
                shares=shares,
                cost_basis=cost_basis,
                state_change_reason="assignment",
                state_changed_at=_utcnow().isoformat(),
            )
            await self._log_state(local.id, local.state, PositionState.SHARES_HELD, "assignment")
        # Persist a synthetic BUY_TO_OPEN order so cycle P&L captures the
        # share cost. Assignment is a corporate action, not a broker order;
        # without this stand-in, _compute_cycle_pnl misses the cost basis.
        if local.current_cycle_id is not None:
            await self._repos.orders.insert(
                Order(
                    account_id=self._account_id,
                    symbol=local.symbol,
                    cycle_id=local.current_cycle_id,
                    order_type=OrderType.BUY_TO_OPEN,
                    contract_symbol=None,
                    strike=None,
                    expiration=None,
                    option_type=None,
                    quantity=shares,
                    limit_price=None,
                    fill_price=strike,
                    status=OrderStatus.FILLED,
                    placed_at=_utcnow(),
                    filled_at=_utcnow(),
                    raw_request=None,
                    raw_response={"synthetic": "assignment"},
                )
            )

    async def _on_csp_expiration(self, local: Position, summary: ReconcileSummary) -> None:
        summary.expirations_processed += 1
        if local.current_cycle_id:
            await self._close_cycle(local.current_cycle_id, CycleOutcome.CSP_EXPIRED, summary)
        if local.id is not None:
            await self._repos.positions.update(
                local.id,
                state=PositionState.IDLE.value,
                shares=0,
                cost_basis=None,
                current_cycle_id=None,
                state_change_reason="csp_expired_worthless",
                state_changed_at=_utcnow().isoformat(),
            )
            await self._log_state(
                local.id, local.state, PositionState.IDLE, "csp_expired_worthless"
            )

    async def _on_cc_expiration(self, local: Position, summary: ReconcileSummary) -> None:
        summary.expirations_processed += 1
        if local.id is not None:
            await self._repos.positions.update(
                local.id,
                state=PositionState.SHARES_HELD.value,
                state_change_reason="cc_expired_worthless",
                state_changed_at=_utcnow().isoformat(),
            )
            await self._log_state(
                local.id, local.state, PositionState.SHARES_HELD, "cc_expired_worthless"
            )

    async def _on_called_away(self, local: Position, summary: ReconcileSummary) -> None:
        # Persist a synthetic SELL_TO_CLOSE order at the CC strike so cycle
        # P&L captures the share leg. Same reason as _on_assignment.
        cc_strike = await self._cycle_cc_strike(local.current_cycle_id)
        # The share-SALE quantity MUST match the share-PURCHASE quantity that
        # assignment recorded (its synthetic BUY_TO_OPEN is sized from
        # _cycle_csp_quantity). If we sized this off local.shares and that count
        # had drifted, the buy and sell legs wouldn't net out and cycle P&L
        # would be wrong. Derive from the same cycle source; fall back to
        # local.shares only when the CSP quantity can't be recovered.
        n_contracts = await self._cycle_csp_quantity(local.current_cycle_id)
        shares_sold = (100 * n_contracts) if n_contracts else local.shares
        # Phantom-loss guard: assignment already recorded the share PURCHASE
        # (synthetic BUY_TO_OPEN at the CSP strike). If we held shares but can't
        # recover the CC strike, we can't record the offsetting SALE — closing
        # the cycle now would book the full cost basis as a loss. Hand it to a
        # human instead of corrupting the books.
        if shares_sold > 0 and cc_strike is None:
            await self._flag_manual_intervention(
                local.symbol,
                "called_away_missing_cc_strike",
                summary,
            )
            return
        summary.called_aways_processed += 1
        await notify(
            "position.called_away",
            f"{local.symbol} called away",
            symbol=local.symbol,
            cycle_id=local.current_cycle_id,
        )
        if local.current_cycle_id is not None and shares_sold > 0 and cc_strike is not None:
            await self._repos.orders.insert(
                Order(
                    account_id=self._account_id,
                    symbol=local.symbol,
                    cycle_id=local.current_cycle_id,
                    order_type=OrderType.SELL_TO_CLOSE,
                    contract_symbol=None,
                    strike=None,
                    expiration=None,
                    option_type=None,
                    quantity=shares_sold,
                    limit_price=None,
                    fill_price=cc_strike,
                    status=OrderStatus.FILLED,
                    placed_at=_utcnow(),
                    filled_at=_utcnow(),
                    raw_request=None,
                    raw_response={"synthetic": "called_away"},
                )
            )
        if local.current_cycle_id:
            await self._close_cycle(local.current_cycle_id, CycleOutcome.CC_CALLED_AWAY, summary)
        if local.id is not None:
            await self._repos.positions.update(
                local.id,
                state=PositionState.IDLE.value,
                shares=0,
                cost_basis=None,
                current_cycle_id=None,
                state_change_reason="called_away",
                state_changed_at=_utcnow().isoformat(),
            )
            await self._log_state(local.id, local.state, PositionState.IDLE, "called_away")

    async def _flag_manual_intervention(
        self,
        symbol: str,
        reason: str,
        summary: ReconcileSummary,
    ) -> None:
        # Dedup BEFORE notify — if the position is already in
        # MANUAL_INTERVENTION, the reconciler may still see the same
        # unknown order on every tick until the cursor advances past it.
        # Without this guard, Discord gets re-pinged each tick for the
        # same underlying event.
        existing = await self._repos.positions.get_by_symbol(self._account_id, symbol)
        if existing is not None and existing.state == PositionState.MANUAL_INTERVENTION:
            return
        summary.manual_interventions += 1
        await notify(
            "position.manual_intervention",
            f"{symbol} flagged for review",
            symbol=symbol,
            reason=reason,
        )
        now = _utcnow()
        if existing is None:
            inserted_id = await self._repos.positions.insert(
                Position(
                    account_id=self._account_id,
                    symbol=symbol,
                    state=PositionState.MANUAL_INTERVENTION,
                    shares=0,
                    state_changed_at=now,
                    state_change_reason=reason,
                )
            )
            await self._log_state(inserted_id, None, PositionState.MANUAL_INTERVENTION, reason)
            return
        if existing.id is not None:
            await self._repos.positions.update_state(
                existing.id,
                PositionState.MANUAL_INTERVENTION,
                reason,
                when=now,
            )
            await self._log_state(
                existing.id, existing.state, PositionState.MANUAL_INTERVENTION, reason
            )

    # -- Helpers --------------------------------------------------------------

    async def _has_inflight_order(self, position: Position) -> bool:
        """True if any order on this position's symbol is still working
        (PENDING/PARTIAL). The stuck-PENDING self-heal uses this to avoid racing
        a genuinely in-flight order — match on symbol only (not strategy_id) so
        the guard stays conservative: if anything is working on the underlying,
        defer the self-heal one tick rather than risk clobbering a transition
        the order stream is about to drive."""
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT 1 FROM orders WHERE symbol = ? AND status IN (?, ?) LIMIT 1",
            (
                position.symbol,
                OrderStatus.PENDING.value,
                OrderStatus.PARTIAL.value,
            ),
        ) as cur:
            return await cur.fetchone() is not None

    async def _set_position_state(
        self,
        position: Position | None,
        symbol: str,
        new_state: PositionState,
        reason: str,
        *,
        cycle_id: int | None = None,
        strategy_id: str | None = None,
    ) -> None:
        now = _utcnow()
        if position is None:
            insert_kwargs: dict[str, Any] = dict(
                account_id=self._account_id,
                symbol=symbol,
                state=new_state,
                shares=0,
                current_cycle_id=cycle_id,
                state_changed_at=now,
                state_change_reason=reason,
            )
            if strategy_id is not None:
                insert_kwargs["strategy_id"] = strategy_id
            inserted_id = await self._repos.positions.insert(Position(**insert_kwargs))
            await self._log_state(inserted_id, None, new_state, reason)
            return
        if position.id is None:
            return
        update_kwargs: dict[str, Any] = {
            "state": new_state.value,
            "state_changed_at": now.isoformat(),
            "state_change_reason": reason,
        }
        if cycle_id is not None and position.current_cycle_id != cycle_id:
            update_kwargs["current_cycle_id"] = cycle_id
        await self._repos.positions.update(position.id, **update_kwargs)
        await self._log_state(position.id, position.state, new_state, reason)

    async def _log_state(
        self,
        position_id: int,
        from_state: PositionState | None,
        to_state: PositionState,
        reason: str,
    ) -> None:
        await self._repos.state_log.insert(
            StateLog(
                position_id=position_id,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                triggered_by=StateLogTrigger.RECONCILER,
                created_at=_utcnow(),
            )
        )

    async def _close_cycle(
        self,
        cycle_id: int,
        outcome: CycleOutcome,
        summary: ReconcileSummary,
    ) -> None:
        cycle = await self._repos.cycles.get(cycle_id)
        if cycle is None or cycle.ended_at is not None:
            return
        pnl = await self._compute_cycle_pnl(cycle_id)
        # Honest labeling: a "*_CLOSED_PROFIT" outcome with negative realized
        # P&L is actually a loss (e.g. stop-loss exit). Relabel accordingly.
        if pnl < 0 and outcome in _PROFIT_TO_LOSS_OUTCOME:
            outcome = _PROFIT_TO_LOSS_OUTCOME[outcome]
        days_held = max((_utcnow() - cycle.started_at).days, 0) if cycle.started_at else None
        capital = cycle.initial_capital_at_risk or 0
        pct = (pnl / capital * 100.0) if capital else None
        await self._repos.cycles.update(
            cycle_id,
            ended_at=_utcnow().isoformat(),
            final_pnl=pnl,
            final_pnl_pct=pct,
            cycle_outcome=outcome.value,
            days_held=days_held,
        )
        summary.cycles_closed += 1
        log_checkpoint(
            "cycle_closed",
            status="ok",
            cycle_id=cycle_id,
            outcome=outcome.value,
            pnl=pnl,
        )
        if pnl < 0:
            await notify(
                "cycle.closed_loss",
                f"{cycle.symbol} cycle closed at a loss",
                cycle_id=cycle_id,
                symbol=cycle.symbol,
                pnl_usd=round(pnl, 2),
                outcome=outcome.value,
                days_held=days_held,
            )

    async def _cycle_csp_quantity(self, cycle_id: int | None) -> int | None:
        """Look up the original CSP fill quantity (in contracts) for a cycle."""
        if cycle_id is None:
            return None
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT quantity FROM orders WHERE cycle_id = ? AND order_type = ? "
            "AND option_type = ? AND status = ? ORDER BY placed_at LIMIT 1",
            (
                cycle_id,
                OrderType.SELL_TO_OPEN.value,
                OptionType.PUT.value,
                OrderStatus.FILLED.value,
            ),
        ) as cur:
            row = await cur.fetchone()
        return int(row["quantity"]) if row and row["quantity"] is not None else None

    async def _cycle_cc_strike(self, cycle_id: int | None) -> float | None:
        """Strike of the most recent filled CC for the cycle (used for the
        synthetic stock SELL_TO_CLOSE on called-away)."""
        if cycle_id is None:
            return None
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT strike FROM orders WHERE cycle_id = ? AND order_type = ? "
            "AND option_type = ? AND status = ? ORDER BY placed_at DESC LIMIT 1",
            (
                cycle_id,
                OrderType.SELL_TO_OPEN.value,
                OptionType.CALL.value,
                OrderStatus.FILLED.value,
            ),
        ) as cur:
            row = await cur.fetchone()
        return float(row["strike"]) if row and row["strike"] is not None else None

    async def _compute_cycle_pnl(self, cycle_id: int) -> float:
        c = await self._repos.db.connect()
        async with c.execute(
            "SELECT order_type, quantity, fill_price, option_type FROM orders "
            "WHERE cycle_id = ? AND status = ? AND fill_price IS NOT NULL",
            (cycle_id, OrderStatus.FILLED.value),
        ) as cur:
            rows = await cur.fetchall()
        pnl = 0.0
        for row in rows:
            qty = row["quantity"] or 0
            price = row["fill_price"] or 0
            ot = row["order_type"]
            option_type = row["option_type"]  # None for stock legs / packages
            if ot in (OrderType.MULTI_LEG_OPEN.value, OrderType.MULTI_LEG_CLOSE.value):
                # fill_price is signed net per share (positive = credit). The
                # qty here is the package quantity; each package is 100 shares
                # per leg ratio. Cash flow = price * qty * 100.
                pnl += price * qty * 100
                continue
            # SELL credits; BUY debits. Options multiplier 100; stock 1.
            sign = 1 if ot in (OrderType.SELL_TO_OPEN.value, OrderType.SELL_TO_CLOSE.value) else -1
            # TICKET-014.5: explicit multiplier map (was a silent `else: 1`).
            if ot in _PNL_OPTION_ORDER_TYPES:
                # SELL_TO_OPEN / BUY_TO_CLOSE — always option trades (100x).
                multiplier = 100
            elif ot in _PNL_STOCK_OR_OPTION_ORDER_TYPES:
                # BUY_TO_OPEN / SELL_TO_CLOSE. Synthetic stock legs from
                # assignment / called-away carry option_type=None → 1x. A leg
                # that DOES carry an option_type is a real option leg (PMCC's
                # long call is the first such case) → 100x. This guard means
                # PMCC's long-call P&L is booked correctly the moment it lands,
                # and the loud log flags it for verification.
                if option_type:
                    multiplier = 100
                    log_checkpoint(
                        "cycle_pnl_option_via_open_close",
                        status="ok",
                        cycle_id=cycle_id,
                        order_type=str(ot),
                        option_type=str(option_type),
                        note="option leg via BUY_TO_OPEN/SELL_TO_CLOSE priced 100x (PMCC long?)",
                    )
                else:
                    multiplier = 1
            else:
                # Unrecognized order type in a cycle's fills. D1: log loudly +
                # safe default (1x under-count, recoverable + visible in
                # /performance) rather than raise — raising here could wedge a
                # cycle close. A new strategy added an order type without
                # teaching the P&L computer its multiplier.
                multiplier = 1
                log_checkpoint(
                    "cycle_pnl_unknown_order_type",
                    status="fail",
                    cycle_id=cycle_id,
                    order_type=str(ot),
                    note="defaulted to 1x multiplier — add this order type to the P&L map",
                )
            pnl += sign * price * qty * multiplier
        return pnl
