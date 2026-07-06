"""TICKET-015 — PMCC close orchestrator.

Two close paths:

  PMCC_BOTH_OPEN → short-call close. Triggers:
      profit: current short mid ≤ (1 − profit_close_pct_short/100) × premium
      time:   short DTE ≤ 1
    Builds a BUY_TO_CLOSE on the short. The reconciler returns the position
    to PMCC_LONG_OPEN; the cycle continues (the long persists).

  PMCC_LONG_OPEN → long roll. Triggers (either fires):
      time:  long DTE < long_roll_dte
      delta: current long delta < long_roll_delta (param absent/0 = off)
    Builds a SELL_TO_CLOSE on the long with trigger_reason="pmcc_roll_dte" /
    "pmcc_roll_delta" (the reconciler matches the "pmcc_roll" prefix and
    closes the cycle as PMCC_LONG_ROLLED); the next IDLE tick opens a fresh
    long (new cycle).

    The delta trigger is the research-backed roll rule: below ~0.70 delta the
    long loses its stock-substitute behavior and theta decay accelerates.
    Delta is computed fresh from broker quotes + BS (same recipe as the
    TICKET-005 wheel delta stop — see wheel_close._current_short_delta), so
    the DTE trigger is the quote-outage backstop: if quotes stay dead the
    roll still fires on time. Delta is only checked from PMCC_LONG_OPEN —
    rolling the long out from under an open short (PMCC_BOTH_OPEN) would
    leave the short naked, so a decayed long waits for the short to close
    first.

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
from core.models import OptionContract, OptionType, Order, OrderType, PositionState
from core.strategies import StrategyDefinition
from data.greeks import fill_greeks
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


async def _current_long_delta(
    broker: Broker, symbol: str, long_order: Order, today: date,
) -> float | None:
    """Current delta of the long call via fresh quotes + BS.

    Same recipe (and same r=0.045 default) as wheel_close._current_short_delta
    so PMCC delta thresholds mean the same thing as wheel ones. Returns None
    when any input is unavailable — the caller logs and skips the trigger for
    this tick; the DTE trigger remains the backstop.
    """
    if (
        long_order.strike is None
        or long_order.expiration is None
        or long_order.contract_symbol is None
    ):
        return None
    try:
        underlying = await broker.get_quote(symbol)
    except Exception:  # noqa: BLE001
        return None
    spot = (
        underlying.mid
        if underlying.mid is not None
        else (underlying.last or underlying.bid or underlying.ask)
    )
    if spot is None:
        return None
    try:
        opt = await broker.get_quote(long_order.contract_symbol)
    except Exception:  # noqa: BLE001
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
        strike=float(long_order.strike),
        expiration=long_order.expiration,
        option_type=long_order.option_type or OptionType.CALL,
        market_price=float(contract_mid),
        today=today,
    )
    if greeks is None:
        return None
    return abs(greeks.delta)


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
        roll_dte = int(params.get("long_roll_dte", 30))
        dte_hit = long_dte < roll_dte

        # Delta roll — checked only when the cheap DTE trigger hasn't already
        # fired (avoids two quote round-trips per tick on healthy positions).
        delta_hit = False
        current_delta: float | None = None
        threshold_raw = params.get("long_roll_delta")
        if not dte_hit and threshold_raw is not None and float(threshold_raw) > 0:
            current_delta = await _current_long_delta(broker, symbol, long_order, today)
            if current_delta is None:
                log_checkpoint(
                    "pmcc_long_delta_unavailable", status="skip",
                    symbol=symbol, long_dte=long_dte,
                )
            elif current_delta < float(threshold_raw):
                delta_hit = True

        if not (dte_hit or delta_hit):
            return None
        long = await _requote(broker, long_order)
        if long is None:
            return None
        if dte_hit:
            reason = "pmcc_roll_dte"
            rationale = (
                f"pmcc_roll_long[{symbol}] dte={long_dte} < {roll_dte} — roll forward"
            )
        else:
            reason = "pmcc_roll_delta"
            rationale = (
                f"pmcc_roll_long[{symbol}] delta={current_delta:.2f} < "
                f"{float(threshold_raw):.2f} (dte={long_dte}) — long lost "
                f"stock-substitute behavior, roll"
            )
        log_checkpoint(
            "pmcc_long_roll_triggered", status="ok", symbol=symbol,
            trigger=reason, long_dte=long_dte, current_delta=current_delta,
        )
        return _build_proposal(
            symbol, long, OrderType.SELL_TO_CLOSE, universe, rationale,
            trigger_reason=reason,
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
