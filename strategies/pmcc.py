"""TICKET-015 — Poor Man's Covered Call (PMCC) entry orchestrator.

Buy a deep-ITM LEAP call (90-180 DTE, ~0.80 delta) as a stock substitute;
sell OTM short-dated calls (7-30 DTE, ~0.27 delta) against it for income.
Net debit, defined risk = the debit. The long appreciates ~like stock; the
short generates premium. Roll the short as it decays; roll the long before
it gets too close to expiration.

CRITICAL RULE (the cc_strike_floor analogue): the short strike must be
STRICTLY ABOVE the long's breakeven (`long.strike + long_debit_per_share`)
so an assigned short is covered by the long AT A PROFIT.

Lifecycle (reconciler-driven, TICKET-015 Phase A):
    IDLE            → propose long  (BUY_TO_OPEN)  → PMCC_LONG_PENDING
    PMCC_LONG_OPEN  → propose short (SELL_TO_OPEN) → PMCC_SHORT_PENDING
    PMCC_BOTH_OPEN  → no entry (closes handled by strategies/pmcc_close.py)

News: the long BUY_TO_OPEN carries news_check_profile="bullish_long" — the
directional bet is news-checked. The short SELL_TO_OPEN is NOT news-checked
(it is covered premium against the long, same reasoning as a wheel CC).

Universe lives in config/universe.yaml (quality names the wheel can't
afford outright — AAPL/MSFT/GOOGL/SPY/QQQ). enabled: false until paper-
validated.

Out of scope (v1): delta-roll of the long (rolls on DTE only), put-side
PMCC, double/ratio diagonals.
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
    PositionState,
)
from core.strategies import StrategyDefinition
from data.chain import ChainFilters, fetch_filtered_chain
from data.ivr import IVRProvider, effective_ivr_min
from db.repo import JSON_FIELDS_BY_TABLE, Repos, _row_to_dict
from strategies.wheel import Proposal, _tier_flags

LONG_NEWS_CHECK_PROFILE = "bullish_long"


def _mid(c: OptionContract) -> float | None:
    if c.bid is None or c.ask is None:
        return None
    m = (c.bid + c.ask) / 2
    return m if m > 0 else None


async def latest_filled_order(
    repos: Repos, cycle_id: int | None, order_type: OrderType,
    *, option_type: OptionType | None = None,
) -> Order | None:
    """Most-recent FILLED order of `order_type` for a cycle. Used to recover
    the long leg (for breakeven) and the latest short leg (for close
    triggers). Mirrors the reconciler's cycle-scoped order queries."""
    if cycle_id is None:
        return None
    sql = (
        "SELECT * FROM orders WHERE cycle_id = ? AND order_type = ? "
        "AND status = ?"
    )
    args: list[Any] = [cycle_id, order_type.value, OrderStatus.FILLED.value]
    if option_type is not None:
        sql += " AND option_type = ?"
        args.append(option_type.value)
    sql += " ORDER BY placed_at DESC LIMIT 1"
    c = await repos.db.connect()
    async with c.execute(sql, tuple(args)) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


def long_breakeven(long_order: Order) -> float | None:
    """Long breakeven = strike + per-share debit paid. The short must sit
    strictly above this so assignment is covered at a profit."""
    if long_order.strike is None or long_order.fill_price is None:
        return None
    return long_order.strike + abs(long_order.fill_price)


async def select_long_call(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    *,
    today: date | None = None,
    record_chain: Any | None = None,
    ivr: IVRProvider | None = None,
) -> OptionContract | None:
    """Deep-ITM LEAP: 90-180 DTE, ~long_delta_target (0.80), the deepest ITM
    contract whose debit fits max_capital_per_position_usd."""
    today = today or date.today()

    if ivr is not None:
        rank = await ivr.iv_rank(symbol)
        ivr_min = effective_ivr_min(ivr, params)  # regime-aware (relaxes when VIX elevated)
        if rank is not None and rank < ivr_min:
            log_checkpoint("pmcc_skip_ivr", status="ok", symbol=symbol, ivr=rank, ivr_min=ivr_min)
            return None

    target = float(params.get("long_delta_target", 0.80))
    cap = float(params.get("max_capital_per_position_usd", 0) or 0)
    filters = ChainFilters(
        dte_min=int(params.get("long_dte_min", 90)),
        dte_max=int(params.get("long_dte_max", 180)),
        delta_min=max(0.50, target - 0.10),
        delta_max=min(0.97, target + 0.15),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    candidates = await fetch_filtered_chain(broker, symbol, "call", filters, today=today)
    if record_chain is not None:
        await record_chain(symbol, "call", candidates)
    if not candidates:
        log_checkpoint("pmcc_no_long_candidates", status="ok", symbol=symbol)
        return None

    # Deepest ITM (highest delta) that fits the capital cap.
    priced = []
    for c in candidates:
        if c.delta is None:
            continue
        m = _mid(c)
        if m is None:
            continue
        debit = m * 100
        if cap > 0 and debit > cap:
            continue
        priced.append((c.delta, c, debit))
    if not priced:
        log_checkpoint(
            "pmcc_long_over_cap", status="ok", symbol=symbol,
            cap=cap, n_filtered=len(candidates),
        )
        return None
    priced.sort(key=lambda t: t[0], reverse=True)  # highest delta first = deepest ITM
    best_delta, best, best_debit = priced[0]
    log_checkpoint(
        "pmcc_long_selected", status="ok", symbol=symbol,
        strike=best.strike, delta=best_delta, debit=best_debit,
        dte=(best.expiration - today).days,
    )
    return best


async def select_short_call(
    broker: Broker,
    symbol: str,
    params: dict[str, Any],
    breakeven: float,
    *,
    today: date | None = None,
    record_chain: Any | None = None,
) -> OptionContract | None:
    """OTM short call: 7-30 DTE, ~short_delta_target (0.27), strike STRICTLY
    above the long breakeven (hard rule). Returns None if nothing qualifies
    above breakeven."""
    today = today or date.today()
    target = float(params.get("short_delta_target", 0.27))
    filters = ChainFilters(
        dte_min=int(params.get("short_dte_min", 7)),
        dte_max=int(params.get("short_dte_max", 30)),
        delta_min=max(0.05, target - 0.12),
        delta_max=min(0.60, target + 0.12),
        open_interest_min=int(params.get("open_interest_min", 0)),
        volume_min=int(params.get("volume_min", 0)),
        bid_ask_spread_max_pct=float(params.get("bid_ask_spread_max_pct", 100.0)),
    )
    candidates = await fetch_filtered_chain(broker, symbol, "call", filters, today=today)
    if record_chain is not None:
        await record_chain(symbol, "call", candidates)
    if not candidates:
        log_checkpoint("pmcc_no_short_candidates", status="ok", symbol=symbol)
        return None

    above = [c for c in candidates if c.strike > breakeven and c.delta is not None and _mid(c) is not None]
    if not above:
        log_checkpoint(
            "pmcc_no_short_above_breakeven", status="ok", symbol=symbol,
            breakeven=breakeven, n_filtered=len(candidates),
        )
        return None
    # Closest to the delta target among the above-breakeven set.
    above.sort(key=lambda c: abs(abs(c.delta) - target))
    best = above[0]
    log_checkpoint(
        "pmcc_short_selected", status="ok", symbol=symbol,
        strike=best.strike, delta=best.delta, breakeven=breakeven,
        dte=(best.expiration - today).days,
    )
    return best


def _build_proposal(
    symbol: str,
    contract: OptionContract,
    order_type: OrderType,
    universe: dict[str, Any],
    rationale: str,
    *,
    news_check_profile: str | None = None,
    trigger_reason: str | None = None,
) -> Proposal:
    needs_screen, needs_human = _tier_flags(symbol, universe)
    return Proposal(
        symbol=symbol,
        contract=contract,
        order_type=order_type,
        quantity=1,  # PMCC is one long + one short per position; sized 1
        rationale=rationale,
        strategy_id="pmcc",
        requires_screen=needs_screen,
        requires_human=needs_human,
        trigger_reason=trigger_reason,
        news_check_profile=news_check_profile,
    )


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    ivr: IVRProvider | None = None,
) -> Proposal | None:
    today = today or date.today()
    if strategy is None:
        log_checkpoint("pmcc_skip_no_strategy", status="fail", symbol=symbol)
        return None
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id="pmcc")
    state = position.state if position else PositionState.IDLE

    if state == PositionState.IDLE:
        long = await select_long_call(broker, symbol, strategy.params, today=today, ivr=ivr)
        if long is None:
            return None
        rationale = (
            f"pmcc_long[{symbol}] strike={long.strike} "
            f"delta={long.delta:.2f} dte={(long.expiration - today).days}"
            if long.delta is not None else f"pmcc_long[{symbol}] strike={long.strike}"
        )
        return _build_proposal(
            symbol, long, OrderType.BUY_TO_OPEN, universe, rationale,
            news_check_profile=LONG_NEWS_CHECK_PROFILE,
        )

    if state == PositionState.PMCC_LONG_OPEN:
        long_order = await latest_filled_order(
            repos, position.current_cycle_id, OrderType.BUY_TO_OPEN,
            option_type=OptionType.CALL,
        )
        if long_order is None:
            log_checkpoint("pmcc_no_long_order", status="fail", symbol=symbol)
            return None
        be = long_breakeven(long_order)
        if be is None:
            log_checkpoint("pmcc_no_breakeven", status="fail", symbol=symbol)
            return None
        short = await select_short_call(broker, symbol, strategy.params, be, today=today)
        if short is None:
            return None
        rationale = (
            f"pmcc_short[{symbol}] strike={short.strike} "
            f"delta={short.delta:.2f} be={be:.2f} dte={(short.expiration - today).days}"
            if short.delta is not None else f"pmcc_short[{symbol}] strike={short.strike} be={be:.2f}"
        )
        # Short sells are NOT news-checked (covered premium) → profile None.
        return _build_proposal(
            symbol, short, OrderType.SELL_TO_OPEN, universe, rationale,
        )

    # PMCC_BOTH_OPEN / pending / closing → no new entry.
    log_checkpoint("pmcc_skip_state", status="ok", symbol=symbol, state=str(state))
    return None


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    ivr: IVRProvider | None = None,
) -> list[Proposal]:
    out: list[Proposal] = []
    for entry in universe["tickers"]:
        p = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy, ivr=ivr,
        )
        if p is not None:
            out.append(p)
    log_checkpoint(
        "pmcc_propose_all", status="ok",
        strategy=strategy.id if strategy else "pmcc", n_proposals=len(out),
    )
    return out
