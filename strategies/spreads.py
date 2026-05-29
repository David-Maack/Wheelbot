"""Vertical credit-spread orchestrator.

Mirrors strategies/wheel.py shape but produces MultiLegProposal — a defined-
risk package the router submits via broker.place_multi_leg_order.

Two directions supported:
  - `bull_put` (sprint 9): short higher-strike put + long lower-strike put.
    Profits when underlying stays flat or rises. Regime gate: csps_allowed.
  - `bear_call` (sprint 10): short lower-strike call + long higher-strike call.
    Profits when underlying stays flat or falls. Regime gate: bear_calls_allowed.

Direction is read from `strategy.params["direction"]` (default `bull_put`
for backwards compatibility with the existing put_spread config). Both
share the close orchestrator, sizing logic, and reconciliation path —
they only differ in selector and risk-gate flag.

Out of scope here (lives in risk/limits.py and execution/router.py):
  - Buying-power floor, concurrent-cap, earnings blackout
  - Order placement, idempotency, retries
  - Reconciliation (multi-leg fill → SPREAD_OPEN)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import (
    ChainSnapshot,
    OptionContract,
    OptionType,
    OrderLeg,
    OrderType,
    PositionState,
)
from core.strategies import StrategyDefinition
from data.ivr import IVRProvider
from db.repo import Repos
from strategies.call_spread_selector import select_bear_call_spread
from strategies.spread_selector import SpreadCandidate, select_bull_put_spread

# Direction labels — match the values used in config.yaml strategy.params.direction.
DIRECTION_BULL_PUT = "bull_put"
DIRECTION_BEAR_CALL = "bear_call"


@dataclass(frozen=True, slots=True)
class MultiLegProposal:
    """A defined-risk multi-leg trade the router submits as a single package.

    Sign convention matches Broker.place_multi_leg_order: positive
    `net_credit_per_spread` = we receive premium. `max_loss_per_spread` is
    always positive ($ at risk per package, before commissions).
    """

    symbol: str
    legs: list[OrderLeg]
    net_credit_per_spread: float    # NET credit per single package, in dollars per share
    max_loss_per_spread: float      # defined max loss per single package, dollars total
    width_dollars: float            # |short - long| strike, in dollars (always positive)
    quantity: int
    rationale: str
    strategy_id: str = "put_spread"
    # Treated as MULTI_LEG_OPEN at submission. Closes are a separate proposal.
    order_type: OrderType = OrderType.MULTI_LEG_OPEN
    requires_screen: bool = False   # tier 2
    requires_human: bool = False    # tier 3
    # Spread direction — drives regime-gate flag selection in risk/limits.py.
    # Defaults to bull_put for backwards-compat with existing put_spread proposals.
    direction: str = DIRECTION_BULL_PUT


def _tier_flags(symbol: str, universe: dict[str, Any]) -> tuple[bool, bool]:
    for entry in universe["tickers"]:
        if entry.symbol.upper() == symbol.upper():
            return (entry.tier == 2, entry.tier == 3)
    return (False, False)


def _make_chain_recorder(
    repos: Repos,
    *,
    cycle_id: int | None = None,
    strategy_id: str | None = None,
) -> Any:
    chain_repo = getattr(repos, "chain_snapshots", None)
    if chain_repo is None:
        async def _noop(symbol: str, side: str, contracts: list[OptionContract]) -> None:
            return None
        return _noop

    async def _record(symbol: str, side: str, contracts: list[OptionContract]) -> None:
        if not contracts:
            return
        underlying_price = next(
            (c.underlying_price for c in contracts if c.underlying_price is not None),
            None,
        )
        await chain_repo.insert(
            ChainSnapshot(
                captured_at=datetime.now(UTC).replace(tzinfo=None),
                symbol=symbol,
                strategy_id=strategy_id,
                side=side,
                underlying_price=underlying_price,
                contracts=[c.model_dump(mode="json") for c in contracts],
                cycle_id=cycle_id,
            )
        )

    return _record


def _build_legs(candidate: SpreadCandidate) -> list[OrderLeg]:
    """Short leg sells to open, long leg buys to open.

    Direction-agnostic: the selector enforces strike relationships
    (puts: short > long; calls: short < long).
    """
    return [
        OrderLeg(
            contract_symbol=candidate.short.occ_symbol,
            underlying=candidate.short.underlying,
            option_type=candidate.short.option_type,
            strike=candidate.short.strike,
            expiration=candidate.short.expiration,
            action=OrderType.SELL_TO_OPEN,
            ratio_qty=1,
        ),
        OrderLeg(
            contract_symbol=candidate.long.occ_symbol,
            underlying=candidate.long.underlying,
            option_type=candidate.long.option_type,
            strike=candidate.long.strike,
            expiration=candidate.long.expiration,
            action=OrderType.BUY_TO_OPEN,
            ratio_qty=1,
        ),
    ]


def _sizing_quantity(
    strategy: StrategyDefinition,
    config: dict[str, Any],
    candidate: SpreadCandidate,
) -> int:
    """Contracts per package given per-strategy capital cap.

    Uses `max_capital_per_spread_usd` from strategy.params if present;
    otherwise falls back to a single contract. Defined risk = max_loss_per_spread.

    Returns 0 when a single spread's defined risk already exceeds the cap — the
    caller must SKIP rather than trade one anyway. Flooring at 1 (the old
    behaviour) silently blew through the per-spread capital limit, which is
    especially dangerous on a small bankroll.
    """
    cap = float(strategy.params.get("max_capital_per_spread_usd", 0) or 0)
    if cap <= 0 or candidate.max_loss_per_spread <= 0:
        # No cap configured (or undefined risk) — default to a single contract.
        return 1
    return int(cap // candidate.max_loss_per_spread)


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
) -> MultiLegProposal | None:
    today = today or date.today()
    account_id = config.get("account", {}).get("id", "primary")
    if strategy is None:
        log_checkpoint("spread_skip_no_strategy", status="fail", symbol=symbol)
        return None
    strategy_id = strategy.id

    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy_id)
    state = position.state if position else PositionState.IDLE

    # Only propose when flat. Open spreads close via separate proposals.
    if state not in (PositionState.IDLE, PositionState.SPREAD_CLOSED):
        log_checkpoint(
            "spread_skip_state",
            status="ok",
            symbol=symbol,
            strategy=strategy_id,
            state=str(state),
        )
        return None

    record = _make_chain_recorder(
        repos,
        cycle_id=position.current_cycle_id if position else None,
        strategy_id=strategy_id,
    )

    direction = str(strategy.params.get("direction", DIRECTION_BULL_PUT))
    if direction == DIRECTION_BEAR_CALL:
        candidate = await select_bear_call_spread(
            broker, symbol, strategy.params, today=today, record_chain=record, ivr=ivr,
        )
    elif direction == DIRECTION_BULL_PUT:
        candidate = await select_bull_put_spread(
            broker, symbol, strategy.params, today=today, record_chain=record, ivr=ivr,
        )
    else:
        log_checkpoint(
            "spread_unknown_direction",
            status="fail",
            symbol=symbol,
            strategy=strategy_id,
            direction=direction,
        )
        return None
    if candidate is None:
        return None

    quantity = _sizing_quantity(strategy, config, candidate)
    if quantity <= 0:
        # One spread's defined risk exceeds the per-spread capital cap — skip
        # rather than over-risk the account (or emit a 0-qty order the broker
        # would reject).
        log_checkpoint(
            "spread_skip_over_cap",
            status="skip",
            symbol=symbol,
            strategy=strategy_id,
            max_loss=candidate.max_loss_per_spread,
            cap=float(strategy.params.get("max_capital_per_spread_usd", 0) or 0),
        )
        return None
    legs = _build_legs(candidate)
    needs_screen, needs_human = _tier_flags(symbol, universe)
    rationale = (
        f"{direction}[{strategy_id}] short={candidate.short.strike} "
        f"long={candidate.long.strike} width={candidate.width_dollars:.2f} "
        f"credit={candidate.net_credit_per_spread:.2f} "
        f"max_loss={candidate.max_loss_per_spread:.2f} qty={quantity}"
    )
    return MultiLegProposal(
        symbol=symbol,
        legs=legs,
        net_credit_per_spread=candidate.net_credit_per_spread,
        max_loss_per_spread=candidate.max_loss_per_spread,
        width_dollars=candidate.width_dollars,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy_id,
        requires_screen=needs_screen,
        requires_human=needs_human,
        direction=direction,
    )


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
    ivr: IVRProvider | None = None,
) -> list[MultiLegProposal]:
    out: list[MultiLegProposal] = []
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy, ivr=ivr,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "spread_propose_all",
        status="ok",
        strategy=strategy.id if strategy else "default",
        n_proposals=len(out),
    )
    return out


# -- Close orchestrator -----------------------------------------------------


async def _open_order_for_position(repos: Repos, position_id: int) -> Any:
    """Most recent FILLED MULTI_LEG_OPEN order for this position's cycle.

    Used to recover the legs (and original credit) we need to construct the
    matching MULTI_LEG_CLOSE proposal.
    """
    c = await repos.db.connect()
    async with c.execute(
        "SELECT * FROM orders WHERE cycle_id = ("
        "SELECT current_cycle_id FROM positions WHERE id = ?"
        ") AND order_type = ? AND status = ? "
        "ORDER BY filled_at DESC LIMIT 1",
        (position_id, OrderType.MULTI_LEG_OPEN.value, "FILLED"),
    ) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    from db.repo import _row_to_dict, JSON_FIELDS_BY_TABLE
    from core.models import Order

    return Order(**_row_to_dict(row, JSON_FIELDS_BY_TABLE["orders"]))


def _flip_action(action: OrderType) -> OrderType:
    """OPEN → CLOSE for the same side."""
    if action == OrderType.SELL_TO_OPEN:
        return OrderType.BUY_TO_CLOSE
    if action == OrderType.BUY_TO_OPEN:
        return OrderType.SELL_TO_CLOSE
    return action


async def _quote_mid(broker: Broker, symbol: str) -> float | None:
    try:
        q = await broker.get_quote(symbol)
    except Exception:
        return None
    if q.mid is not None:
        return q.mid
    return q.last or q.bid or q.ask


async def propose_close_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> MultiLegProposal | None:
    """Build a MULTI_LEG_CLOSE proposal for a SPREAD_OPEN position.

    Triggers (any one is sufficient):
      1. Profit close — current debit-to-close ≤ (1 - profit_close_pct/100) ×
         original credit collected.
      2. Time close — short leg DTE ≤ time_close_dte (force exit before
         gamma/expiration risk dominates).
    """
    today = today or date.today()
    if strategy is None:
        return None
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(
        account_id, symbol, strategy_id=strategy.id
    )
    if position is None or position.state != PositionState.SPREAD_OPEN:
        return None
    if position.id is None:
        return None

    open_order = await _open_order_for_position(repos, position.id)
    if open_order is None or not open_order.raw_request:
        return None
    legs_raw = open_order.raw_request.get("legs") or []
    if not legs_raw:
        return None

    # Reconstruct OrderLegs in CLOSE direction. Infer spread direction from
    # the option_type on any leg (puts → bull_put, calls → bear_call) so the
    # risk gate picks the right regime flag.
    close_legs: list[OrderLeg] = []
    short_leg: OrderLeg | None = None
    direction = DIRECTION_BULL_PUT
    for leg in legs_raw:
        # Each leg dict mirrors OrderLeg.model_dump(mode="json").
        ol = OrderLeg(
            contract_symbol=leg["contract_symbol"],
            underlying=leg["underlying"],
            option_type=leg["option_type"],
            strike=leg["strike"],
            expiration=leg["expiration"],
            action=_flip_action(OrderType(leg["action"])),
            ratio_qty=leg.get("ratio_qty", 1),
        )
        close_legs.append(ol)
        if str(leg["action"]) == OrderType.SELL_TO_OPEN.value:
            short_leg = ol  # remember for DTE check
        if str(leg["option_type"]) == OptionType.CALL.value:
            direction = DIRECTION_BEAR_CALL

    # Re-quote each leg to compute current debit to close.
    leg_mids: list[tuple[OrderLeg, float | None]] = []
    for leg in close_legs:
        mid = await _quote_mid(broker, leg.contract_symbol)
        leg_mids.append((leg, mid))
    if any(mid is None for _, mid in leg_mids):
        log_checkpoint(
            "spread_close_skip_no_quote",
            status="skip",
            symbol=symbol,
            strategy=strategy.id,
        )
        return None

    # Net debit-to-close per share = sum_buy_mid - sum_sell_mid (for the close
    # legs). For a bull put credit spread close: buy short (debit) + sell long
    # (credit). Convention: positive = we pay debit, negative = we still
    # collect a credit (rare, only late in winning trades).
    debit_to_close = 0.0
    for leg, mid in leg_mids:
        assert mid is not None
        if leg.action == OrderType.BUY_TO_CLOSE:
            debit_to_close += mid
        elif leg.action == OrderType.SELL_TO_CLOSE:
            debit_to_close -= mid

    # Original credit collected per package = open_order.fill_price (signed).
    original_credit_per_share = open_order.fill_price or 0.0
    profit_close_pct = float(strategy.params.get("profit_close_pct", 50))
    target_max_debit = (1 - profit_close_pct / 100.0) * original_credit_per_share

    time_close_dte = int(strategy.params.get("time_close_dte", 7))
    short_dte: int | None = None
    if short_leg is not None:
        short_dte = (short_leg.expiration - today).days

    # Stop-loss: close when debit-to-close hits stop_loss_mult × original credit.
    # Research-backed asymmetric defaults: bull_put 2.0, bear_call 1.5
    # (set per strategy in config.yaml). 2.0 is the safe fallback for any
    # spread strategy that didn't set the param.
    stop_loss_mult = float(strategy.params.get("stop_loss_mult", 2.0))
    stop_threshold_debit = stop_loss_mult * original_credit_per_share

    profit_trigger = debit_to_close <= target_max_debit and original_credit_per_share > 0
    time_trigger = short_dte is not None and short_dte <= time_close_dte
    stop_trigger = (
        original_credit_per_share > 0
        and stop_loss_mult > 0
        and debit_to_close >= stop_threshold_debit
    )
    if not (profit_trigger or time_trigger or stop_trigger):
        return None

    quantity = open_order.quantity or 1
    # MultiLegProposal.net_credit_per_spread for a CLOSE: signed net per share
    # we'd receive. For a debit close, this is negative.
    net_close_credit = -debit_to_close
    rationale_parts = []
    if profit_trigger:
        rationale_parts.append(
            f"profit_close at debit={debit_to_close:.2f} ≤ target {target_max_debit:.2f} "
            f"(orig credit={original_credit_per_share:.2f}, pct={profit_close_pct})"
        )
    if stop_trigger:
        rationale_parts.append(
            f"stop_loss at debit={debit_to_close:.2f} ≥ threshold {stop_threshold_debit:.2f} "
            f"(orig credit={original_credit_per_share:.2f}, mult={stop_loss_mult:.1f}×)"
        )
    if time_trigger:
        rationale_parts.append(f"time_close dte={short_dte} ≤ {time_close_dte}")
    rationale = (
        f"{direction}_close[{strategy.id}] " + "; ".join(rationale_parts)
    )

    return MultiLegProposal(
        symbol=symbol,
        legs=close_legs,
        net_credit_per_spread=net_close_credit,
        max_loss_per_spread=0.0,  # closing — no incremental risk
        width_dollars=0.0,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy.id,
        order_type=OrderType.MULTI_LEG_CLOSE,
        direction=direction,
    )


async def propose_all_closes(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[MultiLegProposal]:
    """Walk SPREAD_OPEN positions for this strategy; propose closes."""
    if strategy is None:
        return []
    account_id = config.get("account", {}).get("id", "primary")
    active = await repos.positions.list_active(account_id, strategy_id=strategy.id)
    out: list[MultiLegProposal] = []
    for pos in active:
        if pos.state != PositionState.SPREAD_OPEN:
            continue
        proposal = await propose_close_for_symbol(
            broker, repos, pos.symbol, config, today=today, strategy=strategy,
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint(
        "spread_propose_closes",
        status="ok",
        strategy=strategy.id,
        n_proposals=len(out),
    )
    return out
