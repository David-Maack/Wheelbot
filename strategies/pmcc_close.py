"""TICKET-015 — PMCC close orchestrator.

Two close paths:

  PMCC_BOTH_OPEN → short-call close. Triggers:
      profit: current short mid ≤ (1 − profit_close_pct_short/100) × premium
      time:   short DTE ≤ 1
    Builds a BUY_TO_CLOSE on the short. The reconciler returns the position
    to PMCC_LONG_OPEN; the cycle continues (the long persists).

  PMCC_LONG_OPEN → long roll. Trigger:
      long DTE < long_roll_dte
    Builds a SELL_TO_CLOSE on the long with trigger_reason="pmcc_roll_dte".
    The reconciler closes the cycle as PMCC_LONG_ROLLED; the next IDLE tick
    opens a fresh long (new cycle). v1 rolls on DTE only — delta-roll is a
    future enhancement.

These proposals route through the standard single-leg path. Closes bypass
the entry-window gate and (for the short BUY_TO_CLOSE) the PMCC pending-state
upsert is a no-op — the reconciler drives the state change on the observed
fill.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import OptionContract, OptionType, OrderType, PositionState
from core.strategies import StrategyDefinition
from db.repo import Repos
from strategies.pmcc import _build_proposal, latest_filled_order
from strategies.wheel import Proposal


async def _requote(broker: Broker, order) -> OptionContract | None:
    """Reconstruct an OptionContract for a close from a filled order + a fresh
    quote (the router needs current bid/ask for the limit price)."""
    if order.contract_symbol is None:
        return None
    try:
        q = await broker.get_quote(order.contract_symbol)
    except Exception as exc:  # noqa: BLE001
        log_checkpoint("pmcc_close_no_quote", status="skip",
                       symbol=order.symbol, error=str(exc))
        return None
    if q.bid is None or q.ask is None:
        return None
    return OptionContract(
        underlying=order.symbol,
        occ_symbol=order.contract_symbol,
        strike=order.strike or 0.0,
        expiration=order.expiration or date.today(),
        option_type=order.option_type or OptionType.CALL,
        bid=q.bid, ask=q.ask,
    )


def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    m = (c.bid + c.ask) / 2
    return m if m > 0 else None


async def propose_close_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> Proposal | None:
    today = today or date.today()
    if strategy is None:
        return None
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id="pmcc")
    if position is None:
        return None
    params = strategy.params
    universe = {"tickers": []}  # tier flags irrelevant for closes

    # --- short close (PMCC_BOTH_OPEN) ---
    if position.state == PositionState.PMCC_BOTH_OPEN:
        short_order = await latest_filled_order(
            repos, position.current_cycle_id, OrderType.SELL_TO_OPEN,
            option_type=OptionType.CALL,
        )
        if short_order is None or short_order.fill_price is None:
            return None
        premium = abs(short_order.fill_price)
        short = await _requote(broker, short_order)
        if short is None:
            return None
        current = _mid(short)
        if current is None:
            return None
        profit_pct = float(params.get("profit_close_pct_short", 50)) / 100.0
        profit_hit = current <= (1.0 - profit_pct) * premium
        dte = (short_order.expiration - today).days if short_order.expiration else 99
        time_hit = dte <= int(params.get("short_time_close_dte", 1))
        if not (profit_hit or time_hit):
            return None
        reason = "pmcc_short_profit" if profit_hit else "pmcc_short_time"
        rationale = (
            f"pmcc_close_short[{symbol}] {reason} premium={premium:.2f} "
            f"current={current:.2f} dte={dte}"
        )
        log_checkpoint(
            "pmcc_short_close_triggered", status="ok", symbol=symbol,
            trigger=reason, premium=premium, current=current, dte=dte,
        )
        return _build_proposal(
            symbol, short, OrderType.BUY_TO_CLOSE, universe, rationale,
            trigger_reason=reason,
        )

    # --- long roll (PMCC_LONG_OPEN) ---
    if position.state == PositionState.PMCC_LONG_OPEN:
        long_order = await latest_filled_order(
            repos, position.current_cycle_id, OrderType.BUY_TO_OPEN,
            option_type=OptionType.CALL,
        )
        if long_order is None or long_order.expiration is None:
            return None
        long_dte = (long_order.expiration - today).days
        if long_dte >= int(params.get("long_roll_dte", 30)):
            return None
        long = await _requote(broker, long_order)
        if long is None:
            return None
        rationale = (
            f"pmcc_roll_long[{symbol}] dte={long_dte} < "
            f"{int(params.get('long_roll_dte', 30))} — roll forward"
        )
        log_checkpoint(
            "pmcc_long_roll_triggered", status="ok", symbol=symbol, long_dte=long_dte,
        )
        return _build_proposal(
            symbol, long, OrderType.SELL_TO_CLOSE, universe, rationale,
            trigger_reason="pmcc_roll_dte",
        )

    return None


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[Proposal]:
    if strategy is None:
        return []
    account_id = config.get("account", {}).get("id", "primary")
    active = await repos.positions.list_active(account_id, strategy_id=strategy.id)
    out: list[Proposal] = []
    for pos in active:
        if pos.state not in (PositionState.PMCC_BOTH_OPEN, PositionState.PMCC_LONG_OPEN):
            continue
        p = await propose_close_for_symbol(
            broker, repos, pos.symbol, config, today=today, strategy=strategy,
        )
        if p is not None:
            out.append(p)
    log_checkpoint(
        "pmcc_propose_all_closes", status="ok",
        strategy=strategy.id, n_proposals=len(out),
    )
    return out
