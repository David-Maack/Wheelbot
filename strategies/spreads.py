"""Vertical credit-spread orchestrator (sub-sprint 3).

Mirrors strategies/wheel.py shape but produces MultiLegProposal — a defined-
risk package the router submits via broker.place_multi_leg_order.

Currently implemented: bull put credit spread (short higher-strike put +
long lower-strike put). Iron condors / call spreads slot into the same
shape later.

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
from core.models import ChainSnapshot, OptionContract, OrderLeg, OrderType, PositionState
from core.strategies import StrategyDefinition
from db.repo import Repos
from strategies.spread_selector import SpreadCandidate, select_bull_put_spread


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
    width_dollars: float            # short - long strike, in dollars
    quantity: int
    rationale: str
    strategy_id: str = "put_spread"
    # Treated as MULTI_LEG_OPEN at submission. Closes are a separate proposal.
    order_type: OrderType = OrderType.MULTI_LEG_OPEN
    requires_screen: bool = False   # tier 2
    requires_human: bool = False    # tier 3


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
    """Short the higher strike, long the lower strike (bull put credit spread)."""
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
    """
    cap = float(strategy.params.get("max_capital_per_spread_usd", 0) or 0)
    if cap <= 0 or candidate.max_loss_per_spread <= 0:
        return 1
    contracts = int(cap // candidate.max_loss_per_spread)
    return max(contracts, 1)


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
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

    candidate = await select_bull_put_spread(
        broker, symbol, strategy.params, today=today, record_chain=record
    )
    if candidate is None:
        return None

    quantity = _sizing_quantity(strategy, config, candidate)
    legs = _build_legs(candidate)
    needs_screen, needs_human = _tier_flags(symbol, universe)
    rationale = (
        f"put_spread[{strategy_id}] short={candidate.short.strike} "
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
    )


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[MultiLegProposal]:
    out: list[MultiLegProposal] = []
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe,
            today=today, strategy=strategy,
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
