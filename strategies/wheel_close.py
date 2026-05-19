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
    """Build a BUY_TO_CLOSE proposal if the active short option is at the
    profit threshold. Returns None when no trigger fires."""
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

    threshold_pct = _threshold_pct(strategy, position.state)
    target_max_mid = (1 - threshold_pct / 100.0) * original_premium
    if current_mid > target_max_mid:
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
    rationale = (
        f"wheel_profit_close[{strategy.id}] state={position.state.value if hasattr(position.state, 'value') else position.state} "
        f"mid={current_mid:.2f} ≤ target {target_max_mid:.2f} "
        f"(orig premium={original_premium:.2f}, pct={threshold_pct})"
    )
    log_checkpoint(
        "wheel_profit_close_proposed",
        status="ok",
        symbol=position.symbol,
        strategy=strategy.id,
        contract=short_order.contract_symbol,
        mid=current_mid,
        target=target_max_mid,
        original_premium=original_premium,
        threshold_pct=threshold_pct,
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
    out: list[Proposal] = []
    for pos in active:
        if pos.state not in (PositionState.CSP_OPEN, PositionState.CC_OPEN):
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
