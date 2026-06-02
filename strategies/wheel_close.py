"""Wheel profit-close orchestrator (Sprint 11 — sub-sprint 1).

Locks in CSP / CC profit when the current option mid drops below the
configured threshold, mirroring strategies/spreads.py's close path on the
single-leg side.

Behavior:
  * Active CSP_OPEN positions: when current mid ≤ (1 - csp_profit_close_pct/100)
    × original short premium, propose BUY_TO_CLOSE on the put. Reconciler
    transitions position → IDLE and closes the cycle with CSP_CLOSED_PROFIT.
  * Active CC_OPEN positions: same shape with `cc_profit_close_pct`.
    Reconciler transitions to SHARES_HELD; the cycle stays open for the
    next CC.

Reference premium: most-recent FILLED SELL_TO_OPEN for the position's
current cycle. Captures rolls correctly — a position rolled 3 times is
measured against the latest short's premium, not the cycle's initial one.

Threshold lookup order (per strategy):
  1. csp_profit_close_pct / cc_profit_close_pct (preferred — separate knobs)
  2. profit_close_pct (legacy single param — backwards compat)
  3. default 50

Out of scope (handled elsewhere):
  - Defensive rolls — strategies/roll_orchestrator.py fires on losing positions
  - Order placement / risk gate — execution/router.py + risk/limits.py
  - State transitions on fill — execution/reconciler.py (existing BUY_TO_CLOSE path)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import (
    OptionContract,
    OptionType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PositionState,
)
from core.notify import notify
from core.strategies import StrategyDefinition
from data.greeks import fill_greeks
from db.repo import Repos
from strategies.wheel import Proposal


async def _current_short_delta(
    broker: Broker, position: Position, short_order: Order, today: date
) -> float | None:
    """Compute the *current* |delta| of the short leg via fresh quotes + BS.

    Mirrors `scripts/run_bot._make_roll_evaluator`: spot from the underlying
    quote, contract mid from the option quote, then `fill_greeks` with the
    module defaults (r=0.045, no dividend yield). Using the same defaults as
    the roll evaluator keeps the delta values directly comparable to
    `roll_trigger_delta` — otherwise the two thresholds would mean different
    things and the validation in core.strategies wouldn't catch it.

    Returns None when any input is unavailable; the caller is expected to
    log `wheel_close_delta_unavailable` and skip the trigger for this tick.
    """
    if (
        short_order.strike is None
        or short_order.expiration is None
        or short_order.option_type is None
        or short_order.contract_symbol is None
    ):
        return None
    try:
        underlying = await broker.get_quote(position.symbol)
    except Exception:
        return None
    spot = (
        underlying.mid
        if underlying.mid is not None
        else (underlying.last or underlying.bid or underlying.ask)
    )
    if spot is None:
        return None
    try:
        opt = await broker.get_quote(short_order.contract_symbol)
    except Exception:
        return None
    contract_mid = (
        opt.mid
        if opt.mid is not None
        else (opt.last or (((opt.bid or 0) + (opt.ask or 0)) / 2 if opt.bid and opt.ask else None))
    )
    if contract_mid is None or contract_mid <= 0:
        return None
    greeks = fill_greeks(
        underlying_price=float(spot),
        strike=float(short_order.strike),
        expiration=short_order.expiration,
        option_type=short_order.option_type,
        market_price=float(contract_mid),
        today=today,
    )
    if greeks is None or greeks.delta is None:
        return None
    return abs(greeks.delta)


def _threshold_pct(strategy: StrategyDefinition, state: PositionState) -> float:
    """Return the profit-close percentage threshold for this position's state.

    Lookup order: state-specific key, then legacy single key, then 50% default.
    """
    params = strategy.params
    if state == PositionState.CSP_OPEN:
        return float(
            params.get("csp_profit_close_pct",
                params.get("profit_close_pct", 50))
        )
    if state == PositionState.CC_OPEN:
        return float(
            params.get("cc_profit_close_pct",
                params.get("profit_close_pct", 50))
        )
    raise ValueError(f"profit-close not defined for state {state}")


async def _latest_short_order(repos: Repos, cycle_id: int) -> Order | None:
    """Most recent FILLED SELL_TO_OPEN order on this cycle.

    For a rolled position the cycle has multiple SELL_TO_OPEN orders over
    time; the active short option is the latest one.
    """
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ? "
        "AND order_type = ? AND status = ? "
        "ORDER BY filled_at DESC LIMIT 1",
        (cycle_id, OrderType.SELL_TO_OPEN.value, OrderStatus.FILLED.value),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE
    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


async def _quote_mid(broker: Broker, contract_symbol: str) -> float | None:
    try:
        q = await broker.get_quote(contract_symbol)
    except Exception:
        return None
    if q.mid is not None:
        return q.mid
    if q.bid is not None and q.ask is not None:
        return (q.bid + q.ask) / 2
    return q.last or q.bid or q.ask


async def propose_close_for_position(
    broker: Broker,
    repos: Repos,
    position: Position,
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    has_inflight: bool = False,
    delta_unavailable_counters: dict[int, int] | None = None,
) -> Proposal | None:
    """Build a BUY_TO_CLOSE proposal when any close trigger fires.

    Triggers (independent — any one is sufficient):
      1. Profit close — current mid ≤ (1 - threshold_pct/100) × original premium
      2. Time close (Sprint 13 sub-sprint 2) — short-leg DTE ≤ `time_close_dte`.
         Configured per strategy; omit / set to 0 to disable for this strategy.
         The 21-DTE rule applies cleanly to monthly_wheel; weekly_wheel lives
         entirely inside the gamma window so leaving it unset is the default.
      3. Stop loss (Sprint 14, CSP-only) — current mid ≥ `csp_stop_loss_mult`
         × original premium. Caps single-trade loss on a runaway short put
         before the underlying drops further. Set to 0/None to disable.
         CCs are intentionally excluded — called-away is the wheel's goal,
         not a loss to stop out of; CC management is its own decision.

    Returns None when no trigger fires (or when quote / order data is missing).
    """
    today = today or date.today()
    if strategy is None:
        return None
    if position.state not in (PositionState.CSP_OPEN, PositionState.CC_OPEN):
        return None
    if position.current_cycle_id is None:
        return None

    short_order = await _latest_short_order(repos, position.current_cycle_id)
    if short_order is None or short_order.fill_price is None:
        return None
    if not short_order.contract_symbol or short_order.option_type is None:
        return None
    if short_order.strike is None or short_order.expiration is None:
        return None

    original_premium = float(short_order.fill_price)
    if original_premium <= 0:
        # Defensive — opening a CSP at zero premium would be a data bug, but
        # bailing out keeps us from dividing by it later.
        return None

    current_mid = await _quote_mid(broker, short_order.contract_symbol)
    if current_mid is None:
        log_checkpoint(
            "wheel_close_skip_no_quote",
            status="skip",
            symbol=position.symbol,
            strategy=strategy.id,
            contract=short_order.contract_symbol,
        )
        return None

    # Profit trigger
    threshold_pct = _threshold_pct(strategy, position.state)
    target_max_mid = (1 - threshold_pct / 100.0) * original_premium
    profit_trigger = current_mid <= target_max_mid

    # Time trigger (Sprint 13 sub-sprint 2)
    time_close_dte_raw = strategy.params.get("time_close_dte")
    short_dte = (short_order.expiration - today).days
    time_trigger = (
        time_close_dte_raw is not None
        and int(time_close_dte_raw) > 0
        and short_dte <= int(time_close_dte_raw)
    )

    # Stop-loss trigger (Sprint 14 — CSP only)
    # Read order: csp_stop_loss_mult (preferred) → stop_loss_mult (shared
    # legacy key) → default 2.0. Set to 0 to disable.
    stop_loss_mult: float = 0.0
    stop_threshold_mid: float = 0.0
    stop_trigger = False
    if position.state == PositionState.CSP_OPEN:
        stop_loss_mult = float(
            strategy.params.get(
                "csp_stop_loss_mult",
                strategy.params.get("stop_loss_mult", 2.0),
            )
        )
        if stop_loss_mult > 0:
            stop_threshold_mid = stop_loss_mult * original_premium
            stop_trigger = current_mid >= stop_threshold_mid

    # Delta-stop trigger (TICKET-005) — CSP only (CCs target called-away).
    # Computed AFTER the mid-based checks so the broker quote calls happen only
    # when the cheaper checks haven't already fired. Delta is fetched fresh
    # every tick (no stale persisted delta); see _current_short_delta.
    delta_threshold_raw = strategy.params.get("delta_stop_threshold")
    delta_action = str(strategy.params.get("delta_stop_action", "close")).lower()
    fallback_dte_raw = strategy.params.get("delta_stop_roll_fallback_dte", 7)
    stuck_alert_ticks = int(strategy.params.get("delta_stop_stuck_alert_ticks", 12))
    delta_stop_close_trigger = False
    delta_stop_fallback_trigger = False
    current_delta: float | None = None
    if (
        position.state == PositionState.CSP_OPEN
        and delta_threshold_raw is not None
        and float(delta_threshold_raw) > 0
        and not (profit_trigger or time_trigger or stop_trigger)
    ):
        current_delta = await _current_short_delta(broker, position, short_order, today)
        if current_delta is None:
            # Quote / spot / BS failed. Skip this trigger silently per tick, but
            # tally consecutive failures so a quote that stays dead for ~an hour
            # escalates to Discord (the danger zone is unreadable, which is
            # itself a problem worth a human glance).
            counters = delta_unavailable_counters if delta_unavailable_counters is not None else {}
            key = position.id if position.id is not None else -1
            counters[key] = counters.get(key, 0) + 1
            log_checkpoint(
                "wheel_close_delta_unavailable",
                status="skip",
                symbol=position.symbol,
                strategy=strategy.id,
                contract=short_order.contract_symbol,
                consecutive_failures=counters[key],
                threshold=stuck_alert_ticks,
            )
            if counters[key] == stuck_alert_ticks:
                # Fire ONCE at the threshold — don't spam Discord every tick after.
                await notify(
                    "wheel_close_delta_stuck_unavailable",
                    f"{position.symbol} delta unreadable for "
                    f"{stuck_alert_ticks} consecutive ticks",
                    symbol=position.symbol,
                    strategy=strategy.id,
                    contract=short_order.contract_symbol,
                    state=str(position.state),
                    position_id=position.id,
                )
                log_checkpoint(
                    "wheel_close_delta_stuck_unavailable",
                    status="fail",
                    symbol=position.symbol,
                    strategy=strategy.id,
                    contract=short_order.contract_symbol,
                    consecutive_failures=counters[key],
                )
        else:
            # Reset on any successful read.
            if delta_unavailable_counters is not None and position.id in delta_unavailable_counters:
                delta_unavailable_counters[position.id] = 0
            if current_delta >= float(delta_threshold_raw):
                if delta_action == "close":
                    delta_stop_close_trigger = True
                elif delta_action == "manual":
                    log_checkpoint(
                        "wheel_close_delta_stop_manual",
                        status="ok",
                        symbol=position.symbol,
                        strategy=strategy.id,
                        delta=current_delta,
                        threshold=float(delta_threshold_raw),
                    )
                    if position.id is not None:
                        await repos.positions.update_state(
                            position.id,
                            PositionState.MANUAL_INTERVENTION,
                            f"delta_stop_manual delta={current_delta:.2f}",
                        )
                    return None
                elif delta_action == "roll":
                    # Reconciler's roll scan runs BEFORE this close pass. If it
                    # rolled, an in-flight BTC exists for this (symbol, strategy)
                    # — we're called only when has_inflight is False, so by the
                    # time we get here the roll evaluator either declined
                    # (LET_ASSIGN / CLOSE-without-credit) or no credit-roll was
                    # available. Fall back to close ONLY when we're close to
                    # expiry — the universe's quality gate said we're willing to
                    # hold the shares, so otherwise just defer.
                    if not has_inflight and short_dte <= int(fallback_dte_raw):
                        delta_stop_fallback_trigger = True
                        log_checkpoint(
                            "wheel_close_delta_stop_fallback_chosen",
                            status="ok",
                            symbol=position.symbol,
                            strategy=strategy.id,
                            delta=current_delta,
                            threshold=float(delta_threshold_raw),
                            short_dte=short_dte,
                            fallback_dte=int(fallback_dte_raw),
                        )
                    else:
                        log_checkpoint(
                            "wheel_close_delta_stop_defer_roll",
                            status="ok",
                            symbol=position.symbol,
                            strategy=strategy.id,
                            delta=current_delta,
                            threshold=float(delta_threshold_raw),
                            short_dte=short_dte,
                            has_inflight=has_inflight,
                        )
                        return None
                else:
                    log_checkpoint(
                        "wheel_close_delta_stop_unknown_action",
                        status="fail",
                        symbol=position.symbol,
                        strategy=strategy.id,
                        action=delta_action,
                    )

    if not (
        profit_trigger
        or time_trigger
        or stop_trigger
        or delta_stop_close_trigger
        or delta_stop_fallback_trigger
    ):
        return None

    # Pick the trigger_reason in priority order. profit ranks first because it
    # represents a normal-course exit (most common); stop_loss/delta_stop/time
    # signal a defensive close (each has different post-hoc meaning).
    if profit_trigger:
        trigger_reason = "profit"
    elif stop_trigger:
        trigger_reason = "stop_loss"
    elif delta_stop_close_trigger:
        trigger_reason = "delta_stop_close"
    elif delta_stop_fallback_trigger:
        trigger_reason = "delta_stop_close_fallback"
    else:
        trigger_reason = "time"

    contract = OptionContract(
        underlying=position.symbol,
        occ_symbol=short_order.contract_symbol,
        strike=float(short_order.strike),
        expiration=short_order.expiration,
        option_type=short_order.option_type,
        bid=current_mid,  # set so router/risk-gate liquidity check has a value
        ask=current_mid,
        mid=current_mid,
    )
    state_val = (
        position.state.value if hasattr(position.state, "value") else str(position.state)
    )
    rationale_parts: list[str] = []
    if profit_trigger:
        rationale_parts.append(
            f"profit mid={current_mid:.2f} ≤ target {target_max_mid:.2f} "
            f"(orig {original_premium:.2f}, pct={threshold_pct})"
        )
    if stop_trigger:
        rationale_parts.append(
            f"stop_loss mid={current_mid:.2f} ≥ threshold {stop_threshold_mid:.2f} "
            f"(orig {original_premium:.2f}, mult={stop_loss_mult:.1f}×)"
        )
    if delta_stop_close_trigger or delta_stop_fallback_trigger:
        rationale_parts.append(
            f"delta_stop |delta|={current_delta:.2f} ≥ "
            f"{float(delta_threshold_raw):.2f} (action={delta_action}"
            + (f", fallback dte={short_dte}≤{int(fallback_dte_raw)})" if delta_stop_fallback_trigger else ")")
        )
    if time_trigger:
        rationale_parts.append(
            f"time_close dte={short_dte} ≤ {int(time_close_dte_raw)}"
        )
    rationale = (
        f"wheel_close[{strategy.id}] state={state_val} " + "; ".join(rationale_parts)
    )
    log_checkpoint(
        "wheel_close_proposed",
        status="ok",
        symbol=position.symbol,
        strategy=strategy.id,
        contract=short_order.contract_symbol,
        mid=current_mid,
        trigger_reason=trigger_reason,
        profit_trigger=profit_trigger,
        time_trigger=time_trigger,
        stop_trigger=stop_trigger,
        delta_stop_close_trigger=delta_stop_close_trigger,
        delta_stop_fallback_trigger=delta_stop_fallback_trigger,
        delta=current_delta,
        short_dte=short_dte,
        original_premium=original_premium,
    )
    if trigger_reason == "delta_stop_close":
        log_checkpoint(
            "wheel_close_delta_stop_close",
            status="ok",
            symbol=position.symbol,
            strategy=strategy.id,
            delta=current_delta,
        )
    return Proposal(
        symbol=position.symbol,
        contract=contract,
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=short_order.quantity,
        rationale=rationale,
        strategy_id=strategy.id,
        trigger_reason=trigger_reason,
    )


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    delta_unavailable_counters: dict[int, int] | None = None,
) -> list[Proposal]:
    """Walk active CSP_OPEN / CC_OPEN positions for this strategy; propose
    closes where any trigger fires.

    `delta_unavailable_counters` is process-local state owned by the caller
    (run_bot.main()): a dict[position_id, consecutive_failures] used by the
    delta-stop trigger to escalate a chronically-unreadable quote to Discord.
    Tests can pass a fresh dict; production passes the long-lived one."""
    if strategy is None:
        return []
    account_id = config.get("account", {}).get("id", "primary")
    active = await repos.positions.list_active(account_id, strategy_id=strategy.id)

    # Serialize against the roll evaluator (finding #10). The reconciler's
    # roll-trigger scan runs BEFORE this close pass in the same tick; if it
    # rolled a challenged short it has already placed a BUY_TO_CLOSE (and a
    # SELL_TO_OPEN re-open) that are now PENDING. Proposing a close on the same
    # leg would either collide on the idempotency key (harmless) or, worse, let
    # the roll's re-open win while a stop-loss close was silently subsumed —
    # re-opening a position we wanted flat. Any in-flight order on this
    # (symbol, strategy) means the leg is already being acted on, so skip it
    # and let the pending action resolve first. (Also stops us re-proposing a
    # close while a prior close is still pending.)
    pending = await repos.orders.list_pending(account_id)
    inflight = {
        (o.symbol.upper(), o.strategy_id)
        for o in pending
        if o.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)
    }

    out: list[Proposal] = []
    for pos in active:
        if pos.state not in (PositionState.CSP_OPEN, PositionState.CC_OPEN):
            continue
        pos_inflight = (pos.symbol.upper(), pos.strategy_id) in inflight
        if pos_inflight:
            # In-flight order means the leg is already being acted on. Short-
            # circuit BEFORE calling propose_close_for_position so we don't
            # make broker.get_quote calls for positions we'd skip anyway —
            # important for the per-tick cost on the delta-stop path.
            log_checkpoint(
                "wheel_close_skip_inflight",
                status="skip",
                symbol=pos.symbol,
                strategy=strategy.id,
                reason="order already in flight (roll/close pending)",
            )
            continue
        proposal = await propose_close_for_position(
            broker, repos, pos,
            today=today, strategy=strategy,
            has_inflight=pos_inflight,
            delta_unavailable_counters=delta_unavailable_counters,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "wheel_close_propose_all",
        status="ok",
        strategy=strategy.id,
        n_proposals=len(out),
    )
    return out
