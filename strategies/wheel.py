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
from data.ivr import IVRProvider
from db.repo import Repos
from strategies.cc_selector import select_cc
from strategies.csp_selector import select_csp


@dataclass(frozen=True, slots=True)
class Proposal:
    """A trade the orchestrator wants the router to place. The router (Sprint 4)
    decides idempotency, retries, and dry-run behavior."""

    symbol: str
    contract: OptionContract
    order_type: OrderType  # SELL_TO_OPEN for both CSP and CC at this stage
    quantity: int
    rationale: str  # short human-readable, e.g. "csp 30d delta 0.24 yield 0.32"
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
    quantity: int = 1,
) -> Proposal:
    needs_screen, needs_human = _tier_flags(symbol, universe)
    return Proposal(
        symbol=symbol,
        contract=contract,
        order_type=OrderType.SELL_TO_OPEN,
        quantity=quantity,
        rationale=rationale,
        requires_screen=needs_screen,
        requires_human=needs_human,
    )


def _make_chain_recorder(repos: Repos, *, cycle_id: int | None = None) -> Any:
    """Returns an async callback the selectors hand the post-filter chain to.

    Fail-safe: when the repos bundle has no chain_snapshots attribute (some
    tests use a partial fake), skip the snapshot rather than crashing — the
    snapshot is observability, not load-bearing for the trade itself.
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
                side=side,
                underlying_price=underlying_price,
                contracts=[c.model_dump(mode="json") for c in contracts],
                cycle_id=cycle_id,
            )
        )

    return _record


async def propose_for_symbol(
    broker: Broker,
    repos: Repos,
    symbol: str,
    config: dict[str, Any],
    universe: dict[str, Any],
    ivr: IVRProvider,
    *,
    today: date | None = None,
) -> Proposal | None:
    today = today or date.today()
    account_id = config.get("account", {}).get("id", "primary")
    position = await repos.positions.get_by_symbol(account_id, symbol)
    state = position.state if position else PositionState.IDLE
    record = _make_chain_recorder(repos, cycle_id=position.current_cycle_id if position else None)

    if state == PositionState.IDLE:
        contract = await select_csp(
            broker, symbol, config, universe, ivr, today=today, record_chain=record
        )
        if contract is None:
            return None
        rationale = (
            f"csp dte={(contract.expiration - today).days} "
            f"delta={contract.delta:.2f} strike={contract.strike}"
            if contract.delta is not None
            else f"csp strike={contract.strike}"
        )
        return _build_proposal(symbol, contract, universe, rationale)

    if state == PositionState.SHARES_HELD:
        if position is None or position.cost_basis is None:
            log_checkpoint(
                "wheel_skip_no_cost_basis",
                status="fail",
                symbol=symbol,
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
            config,
            universe,
            today=today,
            record_chain=record,
        )
        if contract is None:
            return None
        rationale = (
            f"cc dte={(contract.expiration - today).days} "
            f"delta={contract.delta:.2f} strike={contract.strike} cb={position.cost_basis}"
            if contract.delta is not None
            else f"cc strike={contract.strike} cb={position.cost_basis}"
        )
        return _build_proposal(symbol, contract, universe, rationale, quantity=contracts)

    log_checkpoint("wheel_skip_state", status="ok", symbol=symbol, state=str(state))
    return None


async def propose_all(
    broker: Broker,
    repos: Repos,
    config: dict[str, Any],
    universe: dict[str, Any],
    ivr: IVRProvider,
    *,
    today: date | None = None,
) -> list[Proposal]:
    out: list[Proposal] = []
    for entry in universe["tickers"]:
        proposal = await propose_for_symbol(
            broker, repos, entry.symbol, config, universe, ivr, today=today
        )
        if proposal is not None:
            out.append(proposal)
    log_checkpoint("wheel_propose_all", status="ok", n_proposals=len(out))
    return out
