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
from core.strategies import StrategyDefinition
from db.repo import Repos
from strategies.wheel import Proposal


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

    if not (profit_trigger or time_trigger or stop_trigger):
        return None

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
        profit_trigger=profit_trigger,
        time_trigger=time_trigger,
        stop_trigger=stop_trigger,
        short_dte=short_dte,
        original_premium=original_premium,
    )
    return Proposal(
        symbol=position.symbol,
        contract=contract,
        order_type=OrderType.BUY_TO_CLOSE,
        quantity=short_order.quantity,
        rationale=rationale,
        strategy_id=strategy.id,
    )


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[Proposal]:
    """Walk active CSP_OPEN / CC_OPEN positions for this strategy; propose
    profit closes where the threshold is met."""
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
        if (pos.symbol.upper(), pos.strategy_id) in inflight:
            log_checkpoint(
                "wheel_close_skip_inflight",
                status="skip",
                symbol=pos.symbol,
                strategy=strategy.id,
                reason="order already in flight (roll/close pending)",
            )
            continue
        proposal = await propose_close_for_position(
            broker, repos, pos, today=today, strategy=strategy,
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
