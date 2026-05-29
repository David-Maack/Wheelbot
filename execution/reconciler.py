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


# When a cycle is closed with a "profit" outcome by the caller but the realized
# P&L is actually negative (e.g. a stop-loss / defensive buyback hardcoded to
# CSP_CLOSED_PROFIT), relabel to the matching loss outcome so the dashboard's
# Outcomes breakdown is honest. The win/loss tally keys off final_pnl sign, so
# it's already correct once P&L is right — this just fixes the label.
_PROFIT_TO_LOSS_OUTCOME = {
    CycleOutcome.CSP_CLOSED_PROFIT: CycleOutcome.CSP_CLOSED_LOSS,
    CycleOutcome.CC_CLOSED_PROFIT: CycleOutcome.CC_CLOSED_LOSS,
    CycleOutcome.SPREAD_CLOSED_PROFIT: CycleOutcome.SPREAD_CLOSED_LOSS,
}


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
            if local_was_pending and now_filled:
                await self._on_fill(local, broker_view, summary)
            elif local_was_pending and now_cancelled:
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
                    CycleOutcome.SPREAD_CLOSED_PROFIT,
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
        legs = (order.raw_request or {}).get("legs") or []
        # Determine width = (max strike) - (min strike) across legs of the same option_type.
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
        broker_by_symbol = {p.symbol.upper(): p for p in broker_positions}
        local_positions = await self._repos.positions.list_all(self._account_id)
        local_by_symbol = {p.symbol.upper(): p for p in local_positions}

        # Detect transitions implied purely by *position state* that fills alone
        # don't surface — assignments and worthless expirations.
        for symbol, local in local_by_symbol.items():
            broker = broker_by_symbol.get(symbol)
            await self._diff_one(symbol, local, broker, summary)

        # Symbols at broker we don't track at all.
        for symbol, broker in broker_by_symbol.items():
            if symbol in local_by_symbol:
                continue
            # Newly discovered position — treat as MANUAL_INTERVENTION rather
            # than guess at state. Spec §10: "do NOT auto-correct."
            await self._flag_manual_intervention(
                broker.symbol,
                f"broker shows {broker.state} for {broker.symbol} but no local row",
                summary,
            )

    async def _diff_one(
        self,
        symbol: str,
        local: Position,
        broker: Position | None,
        summary: ReconcileSummary,
    ) -> None:
        # Case: local says CSP_OPEN. Broker shows shares → assignment.
        # Broker shows nothing → worthless expiration.
        if local.state == PositionState.CSP_OPEN:
            if broker is None or broker.shares == 0 and broker.state not in (
                PositionState.CSP_OPEN,
                PositionState.SHARES_HELD,
            ):
                await self._on_csp_expiration(local, summary)
                return
            if broker.shares >= 100 or broker.state == PositionState.SHARES_HELD:
                await self._on_assignment(local, summary)
                return
            # Still open at broker — nothing to do.
            return

        if local.state == PositionState.CC_OPEN:
            if broker is None or (broker.shares == 0 and broker.state not in (
                PositionState.CC_OPEN,
                PositionState.SHARES_HELD,
            )):
                await self._on_called_away(local, summary)
                return
            if broker.shares > 0 and broker.state == PositionState.SHARES_HELD:
                await self._on_cc_expiration(local, summary)
                return
            return

        if local.state in (PositionState.CSP_PENDING, PositionState.CC_PENDING):
            # Pending means we have an order but no fill yet. No transition here;
            # _on_fill drives it.
            return

        if local.state == PositionState.SPREAD_OPEN:
            # Broker shows nothing for this symbol → both legs OTM at expiry,
            # full credit retained. Anything else is asymmetric (one leg ITM,
            # one OTM, or partial assignment) and needs a human eye.
            if broker is None:
                await self._on_spread_expiration(local, summary)
                return
            # Broker still showing positions on this underlying — could be the
            # spread legs themselves (ok, no action) or shares from assignment
            # of the short put (max-loss path). Conservatively flag any state
            # we can't classify as benign. Stock holdings here are NOT benign:
            # they mean the short put assigned but the long put hasn't been
            # exercised yet — that's defined-risk-realized territory and a
            # human should confirm the close-out path.
            if broker.shares != 0 or broker.state == PositionState.SHARES_HELD:
                await self._flag_manual_intervention(
                    local.symbol,
                    f"spread {local.symbol} shows shares={broker.shares} at broker — "
                    "likely assignment on short leg; review max-loss handling",
                    summary,
                )
            return

        if local.state == PositionState.SPREAD_PENDING:
            # No state-shape transition; _on_fill drives PENDING → OPEN.
            return

        # Anything else: no implied transition from position-shape alone.
        return

    async def _on_spread_expiration(
        self, local: Position, summary: ReconcileSummary
    ) -> None:
        """Both legs OTM at expiry → full credit retained, max profit."""
        summary.expirations_processed += 1
        if local.current_cycle_id:
            await self._close_cycle(
                local.current_cycle_id, CycleOutcome.SPREAD_EXPIRED_PROFIT, summary
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
        shares_sold = local.shares
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
            "SELECT order_type, quantity, fill_price FROM orders "
            "WHERE cycle_id = ? AND status = ? AND fill_price IS NOT NULL",
            (cycle_id, OrderStatus.FILLED.value),
        ) as cur:
            rows = await cur.fetchall()
        pnl = 0.0
        for row in rows:
            qty = row["quantity"] or 0
            price = row["fill_price"] or 0
            ot = row["order_type"]
            if ot in (OrderType.MULTI_LEG_OPEN.value, OrderType.MULTI_LEG_CLOSE.value):
                # fill_price is signed net per share (positive = credit). The
                # qty here is the package quantity; each package is 100 shares
                # per leg ratio. Cash flow = price * qty * 100.
                pnl += price * qty * 100
                continue
            # SELL credits; BUY debits. Options multiplier 100; stock 1.
            sign = 1 if ot in (OrderType.SELL_TO_OPEN.value, OrderType.SELL_TO_CLOSE.value) else -1
            # SELL_TO_OPEN/BUY_TO_CLOSE are option trades (100x); BUY_TO_OPEN/
            # SELL_TO_CLOSE in this codebase are synthetic stock legs (1x)
            # written by assignment / called-away.
            if ot in (OrderType.SELL_TO_OPEN.value, OrderType.BUY_TO_CLOSE.value):
                multiplier = 100
            else:
                multiplier = 1
            pnl += sign * price * qty * multiplier
        return pnl
