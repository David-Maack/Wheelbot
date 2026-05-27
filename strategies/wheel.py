"""Wheel orchestrator.

For each ticker in the universe, look at our local position state and produce
zero or one Proposal:

    state = IDLE          → propose CSP (data/chain → csp_selector)
    state = SHARES_HELD   → propose CC  (data/chain → cc_selector), needs cost basis
    state = anything else → no proposal (managed elsewhere or already in flight)

This module is **stateless and side-effect free** (apart from checkpoint logs).
It does NOT place orders — that's `execution/router.py` (Sprint 4). It does NOT
enforce earnings blackout, BP floor, regime gate, kill switch, position cap, or
concurrent-position cap — those are `risk/limits.py` (Sprint 4) and run *after*
proposals are produced. Selectors here only enforce strike-selection rules
(delta band, DTE band, spread, OI, volume, IVR, CC strike floor).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from core.broker import Broker
from core.checkpoint import log_checkpoint
from core.models import ChainSnapshot, OptionContract, OrderType, PositionState
from core.strategies import StrategyDefinition
from data.ivr import IVRProvider
from db.repo import Repos
from strategies.cc_selector import select_cc
from strategies.csp_selector import select_csp


@dataclass(frozen=True, slots=True)
class Proposal:
    """A trade the orchestrator wants the router to place. The router decides
    idempotency, retries, and dry-run behavior."""

    symbol: str
    contract: OptionContract
    order_type: OrderType  # SELL_TO_OPEN for both CSP and CC at this stage
    quantity: int
    rationale: str  # short human-readable, e.g. "csp 30d delta 0.24 yield 0.32"
    strategy_id: str = "monthly_wheel"
    requires_screen: bool = False  # tier 2
    requires_human: bool = False  # tier 3


def _tier_flags(symbol: str, universe: dict[str, Any]) -> tuple[bool, bool]:
    for entry in universe["tickers"]:
        if entry.symbol.upper() == symbol.upper():
            return (entry.tier == 2, entry.tier == 3)
    return (False, False)


def _build_proposal(
    symbol: str,
    contract: OptionContract,
    universe: dict[str, Any],
    rationale: str,
    strategy_id: str,
    quantity: int = 1,
) -> Proposal:
    needs_screen, needs_human = _tier_flags(symbol, universe)
    return Proposal(
        symbol=symbol,
        contract=contract,
        order_type=OrderType.SELL_TO_OPEN,
        quantity=quantity,
        rationale=rationale,
        strategy_id=strategy_id,
        requires_screen=needs_screen,
        requires_human=needs_human,
    )


def _make_chain_recorder(
    repos: Repos,
    *,
    cycle_id: int | None = None,
    strategy_id: str | None = None,
) -> Any:
    """Returns an async callback the selectors hand the post-filter chain to.

    Fail-safe: when the repos bundle has no chain_snapshots attribute (some
    tests use a partial fake), skip the snapshot rather than crashing.
    """
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


def _config_for_strategy(
    config: dict[str, Any], strategy: StrategyDefinition
) -> dict[str, Any]:
    """Synthesize a config dict where the `wheel` section reflects this
    strategy's params overlaid on the legacy wheel block. Selectors read
    from `wheel`, so this lets one selector serve all wheel-type strategies."""
    base_wheel = config.get("wheel", {}) or {}
    merged_wheel = strategy.merged_wheel_params(base_wheel)
    return {**config, "wheel": merged_wheel}


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    ivr: IVRProvider,
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> Proposal | None:
    today = today or date.today()
    account_id = config.get("account", {}).get("id", "primary")
    strategy_id = strategy.id if strategy is not None else "monthly_wheel"
    position = await repos.positions.get_by_symbol(account_id, symbol, strategy_id=strategy_id)
    state = position.state if position else PositionState.IDLE
    record = _make_chain_recorder(
        repos,
        cycle_id=position.current_cycle_id if position else None,
        strategy_id=strategy_id,
    )
    effective_config = _config_for_strategy(config, strategy) if strategy else config

    if state == PositionState.IDLE:
        contract = await select_csp(
            broker, symbol, effective_config, universe, ivr, today=today, record_chain=record
        )
        if contract is None:
            return None
        rationale = (
            f"csp[{strategy_id}] dte={(contract.expiration - today).days} "
            f"delta={contract.delta:.2f} strike={contract.strike}"
            if contract.delta is not None
            else f"csp[{strategy_id}] strike={contract.strike}"
        )
        return _build_proposal(symbol, contract, universe, rationale, strategy_id)

    if state == PositionState.SHARES_HELD:
        if position is None or position.cost_basis is None:
            log_checkpoint(
                "wheel_skip_no_cost_basis",
                status="fail",
                symbol=symbol,
                strategy=strategy_id,
                state=str(state),
            )
            return None
        # CC contracts are 1 per 100 shares.
        contracts = max(position.shares // 100, 0)
        if contracts == 0:
            return None
        contract = await select_cc(
            broker,
            symbol,
            position.cost_basis,
            effective_config,
            universe,
            today=today,
            record_chain=record,
        )
        if contract is None:
            return None
        rationale = (
            f"cc[{strategy_id}] dte={(contract.expiration - today).days} "
            f"delta={contract.delta:.2f} strike={contract.strike} cb={position.cost_basis}"
            if contract.delta is not None
            else f"cc[{strategy_id}] strike={contract.strike} cb={position.cost_basis}"
        )
        return _build_proposal(
            symbol, contract, universe, rationale, strategy_id, quantity=contracts
        )

    log_checkpoint(
        "wheel_skip_state",
        status="ok",
        symbol=symbol,
        strategy=strategy_id,
        state=str(state),
    )
    return None


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    ivr: IVRProvider,
    *,
    today: date | None = None,
    strategy: StrategyDefinition | None = None,
) -> list[Proposal]:
    out: list[Proposal] = []
    universe_symbols = {t.symbol.upper() for t in universe["tickers"]}

    # 1. Universe tickers — full state dispatch (entries + management).
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe, ivr,
            today=today, strategy=strategy,
        )
        if proposal is not None:
            out.append(proposal)

    # 2. Orphan positions — symbols this strategy owns whose ticker is no
    #    longer in the universe (typically because the operator moved the
    #    symbol to a different strategy mid-cycle, or removed it entirely).
    #    Manage SHARES_HELD only here so the orchestrator can sell covered
    #    calls and complete the wheel cycle. New CSP entries (IDLE state)
    #    stay gated by universe membership — we don't want to re-open new
    #    exposure on a symbol the operator removed.
    #    CSP_OPEN / CC_OPEN closes are already orphan-safe via wheel_close.py.
    if strategy is not None:
        account_id = config.get("account", {}).get("id", "primary")
        active = await repos.positions.list_active(account_id, strategy_id=strategy.id)
        for pos in active:
            if pos.symbol.upper() in universe_symbols:
                continue  # already handled in the loop above
            if pos.state != PositionState.SHARES_HELD:
                continue  # IDLE: no entries; CSP_OPEN/CC_OPEN: handled elsewhere
            log_checkpoint(
                "wheel_orphan_managed",
                status="ok",
                symbol=pos.symbol,
                strategy=strategy.id,
                state=str(pos.state),
            )
            proposal = await propose_for_symbol(
                broker, repos, pos.symbol, config, universe, ivr,
                today=today, strategy=strategy,
            )
            if proposal is not None:
                out.append(proposal)

    log_checkpoint(
        "wheel_propose_all",
        status="ok",
        strategy=strategy.id if strategy else "default",
        n_proposals=len(out),
    )
    return out
